"""
Transcript chunking and semantic retrieval pipeline for VidNote AI RAG.
Groups speech transcript segments into timestamped semantic chunks and
retrieves the most relevant video moments for natural language queries.
"""

import logging
from typing import List, Dict, Any, Optional
from videos.models import format_seconds_to_hhmmss, TranscriptSegment
from .embeddings import generate_embedding
from .vector_store import query_vector_store, VectorStoreError

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Base exception for transcript retrieval failures."""
    pass


class EmptyQueryError(RetrievalError):
    """Raised when search query is blank or invalid."""
    pass


def chunk_transcript_segments(
    segments: List[Any],
    max_chunk_words: int = 80,
    overlap_words: int = 15
) -> List[Dict[str, Any]]:
    """
    Combine granular speech segments into larger cohesive context chunks.
    Ensures that every generated chunk maintains accurate start and end timestamps.
    
    Parameters:
        segments: Iterable of TranscriptSegment model instances or dictionaries.
        max_chunk_words: Maximum approximate word count per chunk.
        overlap_words: Overlap word window between consecutive chunks.
    """
    if not segments:
        return []

    # Sort segments by start_time
    sorted_segments = sorted(
        segments,
        key=lambda s: getattr(s, 'start_time', s.get('start_time', 0.0) if isinstance(s, dict) else 0.0)
    )

    chunks = []
    current_words = []
    current_start = None
    current_end = None
    chunk_index = 0
    video_id = None

    for seg in sorted_segments:
        text = getattr(seg, 'text', seg.get('text', '') if isinstance(seg, dict) else '').strip()
        start = float(getattr(seg, 'start_time', seg.get('start_time', 0.0) if isinstance(seg, dict) else 0.0))
        end = float(getattr(seg, 'end_time', seg.get('end_time', start + 5.0) if isinstance(seg, dict) else start + 5.0))
        
        if video_id is None:
            v_val = getattr(seg, 'video_id', None) or (seg.get('video_id') if isinstance(seg, dict) else None)
            if v_val is not None:
                video_id = int(v_val)

        if not text:
            continue

        seg_words = text.split()

        if current_start is None:
            current_start = start

        current_words.extend(seg_words)
        current_end = end

        # If accumulated enough words, seal chunk
        if len(current_words) >= max_chunk_words:
            chunk_text = " ".join(current_words)
            chunks.append({
                'chunk_id': f"chunk_{video_id or 0}_{chunk_index}",
                'video_id': video_id,
                'start_time': current_start,
                'end_time': current_end,
                'text': chunk_text,
                'word_count': len(current_words)
            })
            chunk_index += 1

            # Keep overlapping tail for continuity
            if overlap_words > 0 and len(current_words) > overlap_words:
                current_words = current_words[-overlap_words:]
                current_start = start  # Start time moves to overlapping segment
            else:
                current_words = []
                current_start = None

    # Append remaining words
    if current_words and current_start is not None:
        chunk_text = " ".join(current_words)
        chunks.append({
            'chunk_id': f"chunk_{video_id or 0}_{chunk_index}",
            'video_id': video_id,
            'start_time': current_start,
            'end_time': current_end,
            'text': chunk_text,
            'word_count': len(current_words)
        })

    return chunks


def search_transcript(
    video_id: int,
    query_text: str,
    top_k: int = 5,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Perform semantic vector similarity search on a specific video's transcript.
    
    Parameters:
        video_id: Primary key of the video to filter by.
        query_text: Natural language user question or query.
        top_k: Maximum number of relevant chunks to retrieve.
        min_score: Minimum similarity score threshold (0.0 to 1.0).
        
    Returns:
        List of relevant transcript chunks with timestamps, score, and text.
    """
    clean_query = str(query_text or '').strip()
    if not clean_query:
        raise EmptyQueryError("Search query cannot be empty.")

    # 1. Generate query embedding
    query_vector = generate_embedding(clean_query)

    # 2. Query Vector Store filtered by video_id
    raw_results = query_vector_store(
        query_embedding=query_vector,
        video_id=video_id,
        top_k=top_k
    )

    # 3. Format and enrich results with human-readable timestamps
    enriched_results = []
    for item in raw_results:
        score = item.get('score', 0.0)
        if score < min_score:
            continue

        start_time = float(item.get('start_time', 0.0))
        end_time = float(item.get('end_time', 0.0))

        item['formatted_start_time'] = format_seconds_to_hhmmss(start_time)
        item['formatted_end_time'] = format_seconds_to_hhmmss(end_time)
        item['formatted_time_range'] = f"{item['formatted_start_time']} - {item['formatted_end_time']}"
        enriched_results.append(item)

    return enriched_results
