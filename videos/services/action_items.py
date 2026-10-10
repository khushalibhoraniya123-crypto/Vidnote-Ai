"""
Action Item Extraction Service for VidNote AI.
Extracts actionable tasks, deliverables, assignments, and follow-ups from
video transcripts (lectures, meetings, presentations) with timestamps,
priorities, and statuses.
"""

import os
import json
import re
import logging
from django.conf import settings
from django.db import transaction
from ..models import Video, TranscriptSegment, ActionItem
from .transcription import process_video_transcription
from .summarization import extract_json_from_response, LLMAPIError

logger = logging.getLogger(__name__)


class ActionItemExtractionError(Exception):
    """Base exception for action item extraction failures."""
    pass


def deduplicate_and_validate_action_items(items_data: list[dict], max_duration: float = None) -> list[dict]:
    """
    Validate and deduplicate extracted action items based on task description similarity and timestamps.
    """
    valid_items = []
    seen_tasks = set()

    for item in items_data:
        try:
            task = str(item.get('task', '')).strip()
            desc = str(item.get('description', '')).strip()
            priority = str(item.get('priority', 'medium')).lower()
            status = str(item.get('status', 'pending')).lower()
            
            raw_ts = item.get('timestamp')
            timestamp = None
            if raw_ts is not None and raw_ts != '':
                timestamp = round(float(raw_ts), 1)
                if timestamp < 0:
                    timestamp = 0.0
                if max_duration and timestamp > max_duration:
                    timestamp = max_duration

            if priority not in ('low', 'medium', 'high'):
                priority = 'medium'

            if status not in ('pending', 'completed'):
                status = 'pending'

            if not task:
                continue

            # Simplified normalized key for deduplication
            task_key = re.sub(r'[^a-zA-Z0-9]', '', task.lower())
            if task_key and task_key in seen_tasks:
                continue

            seen_tasks.add(task_key)
            valid_items.append({
                'task': task[:255],
                'description': desc,
                'priority': priority,
                'status': status,
                'timestamp': timestamp,
            })
        except (ValueError, TypeError):
            continue

    return valid_items


def extract_action_items_heuristic_fallback(segments: list, video_title: str, video_duration: float = None) -> list[dict]:
    """
    Intelligent heuristic action item extractor for meetings and lectures.
    Scans transcript segments for action-oriented phrasing, imperatives, deadlines, and task assignments.
    """
    if not segments:
        return []

    action_patterns = [
        # High priority indicators
        (r'\b(must|urgent|critical|asap|deadline|by today|immediately|due tomorrow|blocker)\b', 'high'),
        # Medium priority indicators
        (r'\b(need to|action item|make sure|ensure|follow up|submit|prepare|review|implement|schedule|send|create|assign)\b', 'medium'),
        # Low priority indicators
        (r'\b(should|consider|look into|optional|nice to have|if time permits|eventually|future)\b', 'low'),
    ]

    action_verbs = r'\b(submit|prepare|review|implement|send|schedule|create|complete|finish|email|write|test|check|verify|fix|update|deploy|present|organize|coordinate|share)\b'

    detected_items = []

    for seg in segments:
        text = seg.text.strip()
        text_lower = text.lower()

        # Check for actionable phrases
        is_actionable = False
        assigned_priority = 'medium'

        for pattern, priority in action_patterns:
            if re.search(pattern, text_lower):
                is_actionable = True
                assigned_priority = priority
                break

        if is_actionable or re.search(action_verbs, text_lower):
            # Formulate a clean task title
            # Extract sentence containing the action
            sentences = re.split(r'[.!?]\s+', text)
            for sentence in sentences:
                s_lower = sentence.lower()
                if any(re.search(pat[0], s_lower) for pat in action_patterns) or re.search(action_verbs, s_lower):
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 10:
                        # Trim for title
                        task_title = clean_sentence
                        if len(task_title) > 80:
                            task_title = task_title[:77] + "..."

                        detected_items.append({
                            'task': task_title,
                            'description': f"Identified from meeting/lecture segment: \"{clean_sentence}\"",
                            'priority': assigned_priority,
                            'status': 'pending',
                            'timestamp': float(seg.start_time),
                        })
                        break

    # If no explicit patterns matched but transcript exists, extract a general review task
    if not detected_items and segments:
        detected_items.append({
            'task': f"Review and summarize takeaways for {video_title}",
            'description': "Actionable follow-up item generated from lecture session.",
            'priority': 'medium',
            'status': 'pending',
            'timestamp': float(segments[0].start_time),
        })

    return deduplicate_and_validate_action_items(detected_items, max_duration=video_duration)


def extract_action_items_via_llm(
    formatted_transcript: str,
    video_title: str,
    video_duration: float = None,
    provider: str = None
) -> list[dict]:
    """
    Extract actionable tasks, assignments, and follow-ups from the transcript using LLMs.
    """
    provider = (provider or getattr(settings, 'LLM_PROVIDER', 'gemini')).lower()

    prompt = f"""
You are an executive assistant and AI task extractor for VidNote AI.
Analyze the following timestamped transcript from the video/meeting "{video_title}" and extract all explicit or implicit ACTION ITEMS, tasks, assignments, deliverables, and follow-up commitments.

TIMESTAMPED TRANSCRIPT:
\"\"\"
{formatted_transcript}
\"\"\"

For each action item:
1. Provide a clear, actionable task title (e.g. "Prepare Q3 financial report", "Review database schema pull request").
2. Provide a detailed context / description of what needs to be done and any mentioned specifics.
3. Identify the exact timestamp (in seconds, float) where this task or assignment was discussed.
4. Assign an appropriate priority level: "high" (urgent/blocker/tight deadline), "medium" (standard core deliverable), or "low" (nice to have/exploratory).
5. Set status as "pending".

Return ONLY a valid JSON array matching this exact schema:
[
  {{
    "task": "Title of the action item",
    "description": "Context and instructions from the meeting/lecture.",
    "priority": "high",
    "status": "pending",
    "timestamp": 45.5
  }}
]
"""

    raw_data = None
    if provider == 'gemini':
        api_key = getattr(settings, 'GEMINI_API_KEY', '') or os.getenv('GEMINI_API_KEY', '')
        if api_key and api_key != 'your_gemini_api_key_here':
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                model_name = getattr(settings, 'LLM_MODEL_NAME', '') or "gemini-2.5-flash"
                res = client.models.generate_content(model=model_name, contents=prompt)
                raw_data = extract_json_from_response(res.text)
            except Exception as e:
                logger.warning(f"Gemini action items extraction failed ({str(e)}). Using fallback.")

    elif provider in ('openai', 'groq'):
        api_key = getattr(settings, 'OPENAI_API_KEY', '') if provider == 'openai' else getattr(settings, 'GROQ_API_KEY', '')
        if api_key and 'your_' not in api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                model_name = getattr(settings, 'LLM_MODEL_NAME', '') or "gpt-4o-mini"
                res = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional task extraction AI assistant. Respond ONLY with valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_data = extract_json_from_response(res.choices[0].message.content)
            except Exception as e:
                logger.warning(f"{provider.capitalize()} action items extraction failed ({str(e)}). Using fallback.")

    # Validate output format
    if isinstance(raw_data, list):
        return deduplicate_and_validate_action_items(raw_data, max_duration=video_duration)
    elif isinstance(raw_data, dict) and 'action_items' in raw_data and isinstance(raw_data['action_items'], list):
        return deduplicate_and_validate_action_items(raw_data['action_items'], max_duration=video_duration)

    return None


def process_action_items_extraction(video_instance: Video, provider: str = None) -> list[ActionItem]:
    """
    Main pipeline to extract, deduplicate, and persist ActionItem records for a video.
    """
    # 1. Ensure transcript is available
    if not video_instance.has_transcript:
        process_video_transcription(video_instance)

    segments = list(video_instance.transcript_segments.all())
    if not segments:
        raise ActionItemExtractionError(f"No transcript segments available for video '{video_instance.title}'.")

    # Format transcript with timestamps
    formatted_lines = [
        f"[{seg.formatted_start_time}] ({seg.start_time:.1f}s): {seg.text}"
        for seg in segments
    ]
    formatted_transcript = "\n".join(formatted_lines)

    # 2. Extract action items via LLM or fallback
    items_data = extract_action_items_via_llm(
        formatted_transcript=formatted_transcript,
        video_title=video_instance.title,
        video_duration=video_instance.duration,
        provider=provider
    )

    if not items_data:
        items_data = extract_action_items_heuristic_fallback(
            segments=segments,
            video_title=video_instance.title,
            video_duration=video_instance.duration
        )

    # 3. Save to database atomically
    with transaction.atomic():
        # Keep completed items or replace items safely
        existing_completed_tasks = set(
            video_instance.action_items.filter(status=ActionItem.STATUS_COMPLETED).values_list('task', flat=True)
        )
        
        # Clear existing pending action items on re-extraction
        video_instance.action_items.filter(status=ActionItem.STATUS_PENDING).delete()

        new_items = []
        for item in items_data:
            task_title = item['task']
            # If already completed by user, keep status completed
            item_status = ActionItem.STATUS_COMPLETED if task_title in existing_completed_tasks else item.get('status', ActionItem.STATUS_PENDING)
            
            # Avoid re-adding if already present
            if not video_instance.action_items.filter(task=task_title).exists():
                new_items.append(
                    ActionItem(
                        video=video_instance,
                        task=task_title,
                        description=item.get('description', ''),
                        priority=item.get('priority', ActionItem.PRIORITY_MEDIUM),
                        status=item_status,
                        timestamp=item.get('timestamp')
                    )
                )

        if new_items:
            ActionItem.objects.bulk_create(new_items)

    return video_instance.action_items.all()
