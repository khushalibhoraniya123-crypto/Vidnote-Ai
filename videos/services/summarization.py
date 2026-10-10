"""
AI Summarization Service for VidNote AI.
Generates structured lecture summaries, key takeaways, topics, and core concepts
using LLM providers (Google Gemini, OpenAI, Groq, or fallback).
"""

import os
import json
import re
import logging
from django.conf import settings
from django.db import transaction
from ..models import Video, Summary
from .transcription import process_video_transcription

logger = logging.getLogger(__name__)

class SummarizationError(Exception):
    """Base exception for summarization failures."""
    pass

class EmptyTranscriptError(SummarizationError):
    """Raised when attempting to summarize an empty transcript."""
    pass

class LLMAPIError(SummarizationError):
    """Raised when the LLM provider API call fails."""
    pass


def chunk_transcript_if_needed(transcript_text: str, max_chars: int = 40000) -> str:
    """
    Safely truncate or window extremely long transcripts to prevent token overflow.
    For standard college lecture videos, 40,000 characters is ~8,000 words.
    """
    if len(transcript_text) <= max_chars:
        return transcript_text
    
    half = max_chars // 2
    return (
        transcript_text[:half] + 
        "\n\n[... middle portion truncated for length ...]\n\n" + 
        transcript_text[-half:]
    )


def extract_json_from_response(raw_text: str) -> dict:
    """
    Extract and parse JSON object from LLM response text, stripping markdown code blocks.
    """
    # 1. Direct JSON parse
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Fallback structure if JSON decoding failed
    logger.warning("Could not parse JSON from LLM response. Using fallback parsing.")
    return {
        "short_summary": text[:400],
        "detailed_summary": text,
        "key_points": [line.strip("- *• ") for line in text.split("\n") if line.strip().startswith(("-", "*", "•"))][:6],
        "main_topics": ["General Lecture Content"],
        "important_concepts": [{"concept": "Core Subject", "explanation": "Key insights detailed in lecture notes."}]
    }


def call_gemini_api(prompt: str, model_name: str = None) -> tuple[dict, str]:
    """
    Call Google Gemini using google-genai SDK.
    """
    api_key = getattr(settings, 'GEMINI_API_KEY', '') or os.getenv('GEMINI_API_KEY', '')
    if not api_key or api_key == 'your_gemini_api_key_here':
        raise LLMAPIError("GEMINI_API_KEY is not configured in .env. Please add your Google Gemini API key.")

    model_name = model_name or getattr(settings, 'LLM_MODEL_NAME', '') or "gemini-2.5-flash"

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return extract_json_from_response(response.text), model_name
    except Exception as e:
        raise LLMAPIError(f"Google Gemini API error: {str(e)}")


def call_openai_api(prompt: str, model_name: str = None) -> tuple[dict, str]:
    """
    Call OpenAI / Groq compatible API.
    """
    provider = getattr(settings, 'LLM_PROVIDER', 'openai')
    if provider == 'groq':
        api_key = getattr(settings, 'GROQ_API_KEY', '') or os.getenv('GROQ_API_KEY', '')
        base_url = "https://api.groq.com/openai/v1"
        default_model = "llama-3.3-70b-versatile"
    else:
        api_key = getattr(settings, 'OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
        base_url = None
        default_model = "gpt-4o-mini"

    if not api_key or 'your_' in api_key:
        raise LLMAPIError(f"API key for {provider.upper()} is not configured in .env.")

    model_name = model_name or getattr(settings, 'LLM_MODEL_NAME', '') or default_model

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert academic summarizer and study notes generator. Respond exclusively in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"} if "gpt" in model_name else None,
            temperature=0.3
        )
        content = response.choices[0].message.content
        return extract_json_from_response(content), model_name
    except Exception as e:
        raise LLMAPIError(f"{provider.capitalize()} API error: {str(e)}")


def generate_structured_summary_fallback(transcript_text: str, video_title: str) -> tuple[dict, str]:
    """
    Intelligent heuristic fallback generator when no external API key is configured.
    Ensures local offline functionality and uninterrupted development/testing.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', transcript_text) if len(s.strip()) > 10]
    
    if not sentences:
        sentences = [transcript_text[:150] or f"Discussion covering {video_title}."]

    short_summary = f"This lecture on '{video_title}' covers key principles and core subject concepts. " + (sentences[0] + "." if sentences else "")
    detailed_summary = (
        f"In this video session focused on {video_title}, several foundational topics and methodologies were reviewed.\n\n"
        + "\n\n".join(sentences[:5]) + "."
    )
    
    key_points = [
        f"Overview and introduction to {video_title}.",
        sentences[0] if len(sentences) > 0 else "Foundational concepts and principles.",
        sentences[1] if len(sentences) > 1 else "Key methodologies and practical applications.",
        sentences[2] if len(sentences) > 2 else "Core problem-solving techniques.",
        "Summary takeaways and review questions."
    ]

    main_topics = [
        f"{video_title} Fundamentals",
        "Theoretical Principles",
        "Implementation & Applications",
        "Analysis & Review"
    ]

    important_concepts = [
        {"concept": f"{video_title} Core Concept", "explanation": sentences[0] if sentences else "Primary subject matter explored in this lecture."},
        {"concept": "Analytical Framework", "explanation": "Structured methodology for understanding the problem domain and solutions."}
    ]

    return {
        "short_summary": short_summary,
        "detailed_summary": detailed_summary,
        "key_points": key_points,
        "main_topics": main_topics,
        "important_concepts": important_concepts
    }, "offline-heuristic-engine"


def generate_summary_from_transcript(
    transcript_text: str,
    video_title: str = "Video Lecture",
    provider: str = None,
    model_name: str = None
) -> tuple[dict, str]:
    """
    Orchestrates sending the transcript to the active LLM provider to produce structured notes.
    """
    clean_text = transcript_text.strip()
    if not clean_text:
        raise EmptyTranscriptError("Transcript is empty. Cannot generate summary.")

    safe_text = chunk_transcript_if_needed(clean_text)
    provider = (provider or getattr(settings, 'LLM_PROVIDER', 'gemini')).lower()

    prompt = f"""
You are an expert AI professor and academic study notes generator for VidNote AI.
Analyze the following transcript from the video titled "{video_title}" and generate comprehensive, well-structured study notes.

TRANSCRIPT:
\"\"\"
{safe_text}
\"\"\"

Return ONLY a valid JSON object matching this exact structure:
{{
  "short_summary": "A concise executive summary (2-3 sentences) capturing the core premise and outcome of the video.",
  "detailed_summary": "A comprehensive multi-paragraph detailed summary providing an in-depth explanation of what was discussed, methodologies, and context.",
  "key_points": [
    "Key takeaway point 1",
    "Key takeaway point 2",
    "Key takeaway point 3",
    "Key takeaway point 4",
    "Key takeaway point 5"
  ],
  "main_topics": [
    "Topic 1",
    "Topic 2",
    "Topic 3"
  ],
  "important_concepts": [
    {{
      "concept": "Concept Name",
      "explanation": "Clear and detailed explanation of this concept from the lecture."
    }}
  ]
}}
"""

    # Dispatch to appropriate provider
    if provider == 'gemini':
        try:
            return call_gemini_api(prompt, model_name=model_name)
        except LLMAPIError as e:
            logger.warning(f"Gemini API failed ({str(e)}). Falling back to heuristic generator.")
            return generate_structured_summary_fallback(clean_text, video_title)

    elif provider in ('openai', 'groq'):
        try:
            return call_openai_api(prompt, model_name=model_name)
        except LLMAPIError as e:
            logger.warning(f"OpenAI/Groq API failed ({str(e)}). Falling back to heuristic generator.")
            return generate_structured_summary_fallback(clean_text, video_title)

    else:
        return generate_structured_summary_fallback(clean_text, video_title)


def process_video_summarization(video_instance: Video, provider: str = None) -> Summary:
    """
    Full pipeline service to generate and save structured summary notes for a video.
    """
    # 1. Ensure transcript is available
    if not video_instance.has_transcript:
        process_video_transcription(video_instance)

    full_transcript = video_instance.full_transcript_text
    if not full_transcript.strip():
        raise EmptyTranscriptError(f"No speech transcript found for video '{video_instance.title}'.")

    # 2. Update status to processing
    video_instance.status = Video.STATUS_PROCESSING
    video_instance.error_message = ""
    video_instance.save(update_fields=['status', 'error_message'])

    try:
        data, model_name = generate_summary_from_transcript(
            transcript_text=full_transcript,
            video_title=video_instance.title,
            provider=provider
        )

        # 3. Store summary in database
        with transaction.atomic():
            summary_obj, created = Summary.objects.update_or_create(
                video=video_instance,
                defaults={
                    'short_summary': data.get('short_summary', ''),
                    'detailed_summary': data.get('detailed_summary', ''),
                    'key_points': data.get('key_points', []),
                    'main_topics': data.get('main_topics', []),
                    'important_concepts': data.get('important_concepts', []),
                    'model_used': model_name or 'ai-summary',
                }
            )

            video_instance.status = Video.STATUS_COMPLETED
            video_instance.error_message = ""
            video_instance.save(update_fields=['status', 'error_message'])

        return summary_obj

    except Exception as exc:
        video_instance.status = Video.STATUS_FAILED
        video_instance.error_message = str(exc)
        video_instance.save(update_fields=['status', 'error_message'])
        raise
