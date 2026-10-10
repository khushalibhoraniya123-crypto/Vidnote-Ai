"""
Important Timestamp & Key Moments Detection service for VidNote AI.
Analyzes speech transcripts with LLMs or heuristic clustering to detect
significant lecture moments, chapter markers, and key concept timestamps.
"""

import os
import json
import re
import logging
from django.conf import settings
from django.db import transaction
from ..models import Video, TranscriptSegment, ImportantMoment
from .transcription import process_video_transcription
from .summarization import extract_json_from_response, LLMAPIError

logger = logging.getLogger(__name__)

class TimestampDetectionError(Exception):
    """Base exception for timestamp detection failures."""
    pass


def deduplicate_and_validate_moments(moments_data: list[dict], max_duration: float = None) -> list[dict]:
    """
    Validate timestamps and remove duplicate or excessively overlapping key moments.
    """
    valid_moments = []
    seen_starts = set()

    for item in moments_data:
        try:
            start = round(float(item.get('start_time', 0.0)), 1)
            end = round(float(item.get('end_time', start + 10.0)), 1)
            title = str(item.get('title', '')).strip()
            desc = str(item.get('description', '')).strip()
            importance = str(item.get('importance', 'high')).lower()

            if importance not in ('critical', 'high', 'medium'):
                importance = 'high'

            if not title:
                continue

            if end < start:
                end = start + 10.0

            if max_duration and start > max_duration:
                start = max(0.0, max_duration - 10.0)
                end = max_duration

            # Deduplication: Check if a moment within 4 seconds start time already exists
            is_dup = any(abs(start - existing_start) < 4.0 for existing_start in seen_starts)
            if not is_dup:
                seen_starts.add(start)
                valid_moments.append({
                    'title': title[:255],
                    'description': desc,
                    'start_time': start,
                    'end_time': end,
                    'importance': importance,
                })
        except (ValueError, TypeError):
            continue

    # Sort chronologically by start time
    valid_moments.sort(key=lambda x: x['start_time'])
    return valid_moments


def detect_moments_heuristic_fallback(segments: list, video_title: str, video_duration: float = None) -> list[dict]:
    """
    Intelligent heuristic timestamp extractor when running offline or without API keys.
    Identifies milestone intervals and keyword cues across transcript segments.
    """
    if not segments:
        return [
            {
                'title': f"Introduction to {video_title}",
                'description': "Opening overview and initial lecture topics.",
                'start_time': 0.0,
                'end_time': min(video_duration or 30.0, 30.0),
                'importance': 'high'
            }
        ]

    moments = []
    
    # 1. First moment: Introduction
    first_seg = segments[0]
    moments.append({
        'title': f"Introduction & Objectives",
        'description': f"Introduction to {video_title} and overview of session goals.",
        'start_time': float(first_seg.start_time),
        'end_time': float(first_seg.end_time) + 5.0,
        'importance': 'high'
    })

    # 2. Keyword cues detection
    keyword_patterns = [
        (r'\b(concept|definition|formula|rule|theorem)\b', "Core Concept Definition", "Detailed explanation of a foundational concept.", "critical"),
        (r'\b(algorithm|method|process|technique|approach)\b', "Methodology & Architecture", "Key technical process or workflow explained.", "critical"),
        (r'\b(example|problem|exercise|code|demo)\b', "Practical Example & Walkthrough", "Practical application or problem walkthrough.", "high"),
        (r'\b(summary|conclusion|takeaway|finally|recap)\b', "Summary & Review", "Review of key conclusions and main takeaways.", "high"),
    ]

    for seg in segments[1:]:
        text_lower = seg.text.lower()
        for pattern, title, desc, importance in keyword_patterns:
            if re.search(pattern, text_lower):
                moments.append({
                    'title': title,
                    'description': f"{desc} ({seg.text[:80]}...)",
                    'start_time': float(seg.start_time),
                    'end_time': float(seg.end_time),
                    'importance': importance
                })
                break

    # If few moments detected, space evenly across segments
    if len(moments) < 3 and len(segments) >= 3:
        mid_idx = len(segments) // 2
        mid_seg = segments[mid_idx]
        moments.append({
            'title': "In-Depth Topic Discussion",
            'description': f"Detailed examination of core lecture material: {mid_seg.text[:60]}...",
            'start_time': float(mid_seg.start_time),
            'end_time': float(mid_seg.end_time) + 10.0,
            'importance': 'high'
        })

    return deduplicate_and_validate_moments(moments, max_duration=video_duration)


def detect_important_moments_via_llm(
    formatted_transcript: str,
    video_title: str,
    video_duration: float = None,
    provider: str = None
) -> list[dict]:
    """
    Send formatted transcript with timestamps to LLM to detect key moments.
    """
    provider = (provider or getattr(settings, 'LLM_PROVIDER', 'gemini')).lower()

    prompt = f"""
You are an expert video indexing AI for VidNote AI.
Analyze the following timestamped transcript from the video "{video_title}" and identify 3 to 6 of the most important, high-value moments/chapters.

TIMESTAMPED TRANSCRIPT:
\"\"\"
{formatted_transcript}
\"\"\"

For each important moment:
1. Identify the exact start_time and end_time in seconds based on the transcript timestamps.
2. Provide a clear, concise title.
3. Provide a brief explanation of why this moment is significant.
4. Assign an importance level ("critical", "high", or "medium").

Return ONLY a valid JSON array matching this structure:
[
  {{
    "title": "Title of Key Moment",
    "description": "Short explanation of why this moment is significant to the student.",
    "start_time": 12.5,
    "end_time": 45.0,
    "importance": "critical"
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
                logger.warning(f"Gemini timestamp detection failed ({str(e)}). Using fallback.")

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
                        {"role": "system", "content": "You are a timestamp detection assistant. Output ONLY valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_data = extract_json_from_response(res.choices[0].message.content)
            except Exception as e:
                logger.warning(f"{provider.capitalize()} timestamp detection failed ({str(e)}). Using fallback.")

    # Validate output format
    if isinstance(raw_data, list):
        return deduplicate_and_validate_moments(raw_data, max_duration=video_duration)
    elif isinstance(raw_data, dict) and 'moments' in raw_data and isinstance(raw_data['moments'], list):
        return deduplicate_and_validate_moments(raw_data['moments'], max_duration=video_duration)

    return None


def process_important_moments_detection(video_instance: Video, provider: str = None) -> list[ImportantMoment]:
    """
    Full pipeline to detect, deduplicate, and persist ImportantMoment records in the database.
    """
    # 1. Ensure transcript is available
    if not video_instance.has_transcript:
        process_video_transcription(video_instance)

    segments = list(video_instance.transcript_segments.all())
    if not segments:
        raise TimestampDetectionError(f"No transcript segments found for video '{video_instance.title}'.")

    # Format transcript with timestamps
    formatted_lines = [
        f"[{seg.formatted_start_time} - {seg.formatted_end_time}] ({seg.start_time:.1f}s): {seg.text}"
        for seg in segments
    ]
    formatted_transcript = "\n".join(formatted_lines)

    # 2. Detect moments via LLM or fallback
    moments_data = detect_important_moments_via_llm(
        formatted_transcript=formatted_transcript,
        video_title=video_instance.title,
        video_duration=video_instance.duration,
        provider=provider
    )

    if not moments_data:
        moments_data = detect_moments_heuristic_fallback(
            segments=segments,
            video_title=video_instance.title,
            video_duration=video_instance.duration
        )

    # 3. Save to database atomically
    with transaction.atomic():
        # Clear existing moments to avoid duplicates on re-detection
        video_instance.important_moments.all().delete()

        moment_objects = [
            ImportantMoment(
                video=video_instance,
                title=item['title'],
                description=item.get('description', ''),
                start_time=item['start_time'],
                end_time=item['end_time'],
                importance=item.get('importance', ImportantMoment.IMPORTANCE_HIGH)
            )
            for item in moments_data
        ]

        if moment_objects:
            ImportantMoment.objects.bulk_create(moment_objects)

    return video_instance.important_moments.all()
