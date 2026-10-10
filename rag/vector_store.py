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


import numpy as np

# In-memory numpy vector store fallback for lightweight serverless environments
class InMemoryVectorCollection:
    def __init__(self, name: str = COLLECTION_NAME):
        self.name = name
        self.records = []  # List of dicts with 'id', 'embedding', 'document', 'metadata'

    def delete(self, where: Optional[Dict[str, Any]] = None):
        if not where:
            self.records = []
            return
        v_id = where.get("video_id")
        if v_id is not None:
            self.records = [r for r in self.records if r['metadata'].get('video_id') != int(v_id)]

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        for i in range(len(ids)):
            self.records.append({
                'id': ids[i],
                'embedding': np.array(embeddings[i], dtype=np.float32),
                'document': documents[i],
                'metadata': metadatas[i]
            })

    def query(self, query_embeddings: List[List[float]], n_results: int = 5, where: Optional[Dict[str, Any]] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.records or not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_vec = np.array(query_embeddings[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        v_filter = where.get("video_id") if where else None

        candidates = []
        for r in self.records:
            if v_filter is not None and r['metadata'].get('video_id') != int(v_filter):
                continue
            r_vec = r['embedding']
            r_norm = np.linalg.norm(r_vec)
            if r_norm == 0:
                r_norm = 1.0
            # Cosine similarity -> cosine distance (1 - cos_sim)
            sim = float(np.dot(q_vec, r_vec) / (q_norm * r_norm))
            dist = max(0.0, min(2.0, 1.0 - sim))
            candidates.append((dist, r))

        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:n_results]

        ids = [c[1]['id'] for c in top_candidates]
        docs = [c[1]['document'] for c in top_candidates]
        metas = [c[1]['metadata'] for c in top_candidates]
        distances = [c[0] for c in top_candidates]

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances]
        }

_IN_MEMORY_COLLECTION = InMemoryVectorCollection()


def get_chroma_client():
    """
    Initialize and return ChromaDB Persistent Client, or None if unavailable.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT

    persist_dir = getattr(settings, 'CHROMA_PERSIST_DIRECTORY', None)
    if not persist_dir:
        persist_dir = os.path.join(settings.BASE_DIR, 'media', 'chroma_db')
    
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        pass

    try:
        import chromadb
        from chromadb.config import Settings
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
        return _CHROMA_CLIENT
    except Exception as e:
        logger.info(f"Persistent ChromaDB client unavailable ({str(e)}). Testing ephemeral client...")
        try:
            import chromadb
            _CHROMA_CLIENT = chromadb.EphemeralClient()
            return _CHROMA_CLIENT
        except Exception:
            return None


def get_or_create_collection():
    """
    Get or create vector collection (ChromaDB or high-performance NumPy in-memory collection).
    """
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    client = get_chroma_client()
    if client is not None:
        try:
            _CHROMA_COLLECTION = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "VidNote AI video transcript segment embeddings", "hnsw:space": "cosine"}
            )
            return _CHROMA_COLLECTION
        except Exception as e:
            logger.warning(f"Chroma collection access error ({str(e)}). Using NumPy vector fallback.")

    _CHROMA_COLLECTION = _IN_MEMORY_COLLECTION
    return _CHROMA_COLLECTION



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
