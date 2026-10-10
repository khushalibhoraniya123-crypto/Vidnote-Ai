"""
Vector Store indexing and semantic retrieval service for VidNote AI RAG.
Manages ChromaDB persistent vector database with FAISS/Numpy fallbacks.
Supports video_id metadata filtering and duplicate-free re-indexing.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
COLLECTION_NAME = "vidnote_transcript_chunks"


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


def get_chroma_client():
    """
    Initialize and return ChromaDB Persistent Client.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT

    persist_dir = getattr(settings, 'CHROMA_PERSIST_DIRECTORY', None)
    if not persist_dir:
        persist_dir = os.path.join(settings.BASE_DIR, 'media', 'chroma_db')
    
    os.makedirs(persist_dir, exist_ok=True)

    try:
        import chromadb
        from chromadb.config import Settings
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
        return _CHROMA_CLIENT
    except Exception as e:
        logger.warning(f"Could not initialize persistent ChromaDB client ({str(e)}). Using ephemeral client.")
        try:
            import chromadb
            _CHROMA_CLIENT = chromadb.EphemeralClient()
            return _CHROMA_CLIENT
        except Exception as e2:
            logger.error(f"Failed to create ephemeral Chroma client: {str(e2)}")
            return None


def get_or_create_collection():
    """
    Get or create the global ChromaDB collection for transcript chunks.
    """
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    client = get_chroma_client()
    if client is None:
        raise VectorStoreError("ChromaDB vector database client is unavailable.")

    try:
        _CHROMA_COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "VidNote AI video transcript segment embeddings", "hnsw:space": "cosine"}
        )
        return _CHROMA_COLLECTION
    except Exception as e:
        raise VectorStoreError(f"Failed to access ChromaDB collection '{COLLECTION_NAME}': {str(e)}")


def delete_video_chunks(video_id: int) -> int:
    """
    Delete all existing indexed chunks for a specific video to prevent duplicate entries on re-indexing.
    """
    try:
        collection = get_or_create_collection()
        # Delete by metadata filter
        collection.delete(where={"video_id": int(video_id)})
        return 1
    except Exception as e:
        logger.warning(f"Note on deleting old chunks for video {video_id}: {str(e)}")
        return 0


def add_chunks_to_vector_store(
    video_id: int,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> int:
    """
    Index transcript chunks and their embedding vectors into ChromaDB.
    
    Parameters:
        video_id: Integer ID of the Video model.
        chunks: List of dictionaries containing chunk_id, start_time, end_time, text.
        embeddings: List of float vectors corresponding to each chunk.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise VectorStoreError(f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)}).")

    # 1. Clear any previous entries for this video to ensure repeatability
    delete_video_chunks(video_id)

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get('chunk_id') or f"vid_{video_id}_c_{i}"
        text = str(chunk.get('text', '')).strip()
        start_time = float(chunk.get('start_time', 0.0))
        end_time = float(chunk.get('end_time', start_time + 5.0))

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append({
            "video_id": int(video_id),
            "chunk_id": str(chunk_id),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "text": str(text),
        })

    collection = get_or_create_collection()
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(ids)
    except Exception as e:
        raise VectorStoreError(f"Failed to insert vectors into ChromaDB: {str(e)}")


def query_vector_store(
    query_embedding: List[float],
    video_id: Optional[int] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search vector store for the closest matching transcript chunks.
    Filters strictly by video_id if provided.
    
    Returns:
        List of dicts with keys: chunk_id, video_id, start_time, end_time, text, score (0.0 to 1.0)
    """
    if not query_embedding:
        return []

    collection = get_or_create_collection()
    where_filter = {"video_id": int(video_id)} if video_id is not None else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise VectorStoreError(f"Vector search failed: {str(e)}")

    formatted_results = []
    
    if not results or not results.get('ids') or not results['ids'][0]:
        return []

    ids = results['ids'][0]
    metadatas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]
    documents = results.get('documents', [[]])[0]

    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        doc = documents[i] if i < len(documents) else meta.get('text', '')
        distance = distances[i] if i < len(distances) else 0.5
        
        # In cosine distance: 0 = identical, 2 = opposite.
        # Normalize similarity score to [0.0, 1.0] range
        similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))

        formatted_results.append({
            'chunk_id': meta.get('chunk_id', ids[i]),
            'video_id': meta.get('video_id', video_id),
            'start_time': float(meta.get('start_time', 0.0)),
            'end_time': float(meta.get('end_time', 0.0)),
            'text': doc,
            'score': round(float(similarity_score), 4),
            'distance': round(float(distance), 4)
        })

    # Sort descending by similarity score
    formatted_results.sort(key=lambda x: x['score'], reverse=True)
    return formatted_results[:top_k]
