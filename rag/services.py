"""
High-level RAG Orchestration Service for VidNote AI.
Coordinates transcript chunking, embedding generation, vector indexing,
and semantic question-answering search for video transcripts.
"""

import os
import logging
from typing import List, Dict, Any
from django.conf import settings
from django.db import transaction
from django.contrib.auth.models import User
from videos.models import Video, TranscriptSegment
from videos.services.transcription import process_video_transcription
from .models import ChatMessage
from .embeddings import generate_embeddings
from .vector_store import add_chunks_to_vector_store, delete_video_chunks, VectorStoreError
from .retrieval import chunk_transcript_segments, search_transcript, RetrievalError, EmptyQueryError

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I could not find sufficient information about this in the uploaded video."


class RAGServiceError(Exception):
    """Base exception for RAG orchestration service."""
    pass


class VideoNotTranscribedError(RAGServiceError):
    """Raised when indexing a video that has no speech transcripts."""
    pass


def index_video_transcript(video_instance: Video) -> Dict[str, Any]:
    """
    Complete indexing pipeline for a video:
    1. Validates or extracts transcript segments.
    2. Groups segments into timestamped semantic chunks.
    3. Computes vector embeddings.
    4. Persists vectors into ChromaDB vector store with video_id isolation.
    """
    if not video_instance.has_transcript:
        process_video_transcription(video_instance)

    segments = list(video_instance.transcript_segments.all())
    if not segments:
        raise VideoNotTranscribedError(f"Video '{video_instance.title}' has no transcript segments to index.")

    # 1. Chunk transcript segments
    chunks = chunk_transcript_segments(segments, max_chunk_words=80, overlap_words=15)
    if not chunks:
        # If segments were very short, treat each as a single chunk
        chunks = [
            {
                'chunk_id': f"chunk_{video_instance.pk}_{i}",
                'video_id': video_instance.pk,
                'start_time': float(seg.start_time),
                'end_time': float(seg.end_time),
                'text': seg.text.strip(),
                'word_count': len(seg.text.split())
            }
            for i, seg in enumerate(segments)
        ]

    # Set video_id explicitly across all chunks
    for chk in chunks:
        chk['video_id'] = video_instance.pk

    # 2. Extract texts for batch embedding
    texts = [chk['text'] for chk in chunks]

    # 3. Generate dense vector embeddings
    embeddings = generate_embeddings(texts)

    # 4. Save into ChromaDB with video_id filter
    indexed_count = add_chunks_to_vector_store(
        video_id=video_instance.pk,
        chunks=chunks,
        embeddings=embeddings
    )

    logger.info(f"Successfully indexed {indexed_count} transcript chunks for video {video_instance.pk} ('{video_instance.title}').")

    return {
        'video_id': video_instance.pk,
        'title': video_instance.title,
        'chunks_count': indexed_count,
        'status': 'indexed'
    }


def semantic_search_video(
    video_instance: Video,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Execute semantic search over a specific video's transcript.
    Auto-indexes the video if not previously indexed in the vector store.
    """
    clean_query = str(query or '').strip()
    if not clean_query:
        raise EmptyQueryError("Please provide a valid question or search query.")

    if not video_instance.has_transcript:
        raise VideoNotTranscribedError(f"Cannot search video '{video_instance.title}' without a transcript.")

    try:
        results = search_transcript(
            video_id=video_instance.pk,
            query_text=clean_query,
            top_k=top_k,
            min_score=min_score
        )
    except Exception:
        # If collection is empty or not indexed yet, trigger indexing and retry search
        index_video_transcript(video_instance)
        results = search_transcript(
            video_id=video_instance.pk,
            query_text=clean_query,
            top_k=top_k,
            min_score=min_score
        )

    return results


def generate_grounded_answer_via_llm(
    video_title: str,
    question: str,
    context_chunks: List[Dict[str, Any]],
    provider: str = None
) -> str:
    """
    Query LLM with strict grounding constraints.
    Returns NOT_FOUND_MESSAGE if no sufficient evidence is found in the context.
    """
    if not context_chunks:
        return NOT_FOUND_MESSAGE

    provider = (provider or getattr(settings, 'LLM_PROVIDER', 'gemini')).lower()

    # Format excerpts with timestamps
    context_lines = []
    for c in context_chunks:
        start_fmt = c.get('formatted_start_time') or f"{c.get('start_time', 0.0):.1f}s"
        end_fmt = c.get('formatted_end_time') or f"{c.get('end_time', 0.0):.1f}s"
        context_lines.append(f"[{start_fmt} - {end_fmt}] {c.get('text', '')}")

    formatted_context = "\n\n".join(context_lines)

    system_prompt = (
        f"You are an AI teaching assistant for VidNote AI answering questions about the video \"{video_title}\".\n"
        "CRITICAL RULES:\n"
        "1. Answer the user's question accurately and concisely using ONLY the provided transcript excerpts below.\n"
        "2. If the provided excerpts do NOT contain enough information to answer the question, you MUST respond EXACTLY with:\n"
        f"\"{NOT_FOUND_MESSAGE}\"\n"
        "3. DO NOT invent, assume, extrapolate, or hallucinate facts not in the context.\n"
        "4. Include timestamp references (e.g. [00:01:23]) when citing facts from the video.\n"
        "5. Keep the tone helpful, clear, and direct."
    )

    user_prompt = (
        f"TRANSCRIPT EXCERPTS:\n\"\"\"\n{formatted_context}\n\"\"\"\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    # 1. Gemini
    if provider == 'gemini':
        api_key = getattr(settings, 'GEMINI_API_KEY', '') or os.getenv('GEMINI_API_KEY', '')
        if api_key and api_key != 'your_gemini_api_key_here':
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                model_name = getattr(settings, 'LLM_MODEL_NAME', '') or "gemini-2.5-flash"
                res = client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\n{user_prompt}"
                )
                if res.text and res.text.strip():
                    return res.text.strip()
            except Exception as e:
                logger.warning(f"Gemini RAG answer generation failed ({str(e)}). Using fallback.")

    # 2. OpenAI / Groq
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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                if res.choices and res.choices[0].message.content:
                    return res.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"{provider.capitalize()} RAG answer generation failed ({str(e)}). Using fallback.")

    # 3. Grounded Heuristic Synthesis Fallback (offline / testing)
    top_chunk = context_chunks[0]
    top_score = top_chunk.get('score', 0.0)

    # Check if query keywords overlap with top chunk
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    chunk_words = set(w.lower() for w in top_chunk.get('text', '').split())
    overlap = q_words.intersection(chunk_words)

    if len(overlap) == 0 or top_score < 0.2:
        return NOT_FOUND_MESSAGE

    start_str = top_chunk.get('formatted_start_time', '00:00:00')
    return f"According to the lecture at [{start_str}]: {top_chunk.get('text', '').strip()}"


def answer_video_question(
    video_instance: Video,
    user: User,
    question_text: str,
    provider: str = None
) -> ChatMessage:
    """
    Full end-to-end RAG question answering pipeline:
    1. Records user ChatMessage in database.
    2. Performs semantic search over video's transcript in ChromaDB.
    3. Feeds grounded context to LLM with strict hallucination guards.
    4. Records assistant ChatMessage with source timestamps in database.
    """
    clean_question = str(question_text or '').strip()
    if not clean_question:
        raise EmptyQueryError("Question cannot be empty.")

    with transaction.atomic():
        # 1. Save user query message
        user_msg = ChatMessage.objects.create(
            video=video_instance,
            user=user,
            role=ChatMessage.ROLE_USER,
            content=clean_question
        )

        # 2. Retrieve relevant transcript chunks
        retrieved_chunks = semantic_search_video(
            video_instance=video_instance,
            query=clean_question,
            top_k=4,
            min_score=0.1
        )

        # 3. Format sources payload
        sources = [
            {
                'chunk_id': c.get('chunk_id'),
                'start_time': c.get('start_time'),
                'end_time': c.get('end_time'),
                'formatted_start_time': c.get('formatted_start_time', '00:00:00'),
                'formatted_end_time': c.get('formatted_end_time', '00:00:00'),
                'formatted_time_range': c.get('formatted_time_range', ''),
                'text': c.get('text', '')[:250],
                'score': c.get('score', 0.0)
            }
            for c in retrieved_chunks
        ]

        # 4. Generate grounded LLM response
        answer_text = generate_grounded_answer_via_llm(
            video_title=video_instance.title,
            question=clean_question,
            context_chunks=retrieved_chunks,
            provider=provider
        )

        # 5. Save and return assistant message
        assistant_msg = ChatMessage.objects.create(
            video=video_instance,
            user=user,
            role=ChatMessage.ROLE_ASSISTANT,
            content=answer_text,
            sources=sources if answer_text != NOT_FOUND_MESSAGE else []
        )

    return assistant_msg

