"""
Embeddings generation service for VidNote AI RAG.
Produces high-dimensional vector representations using Sentence Transformers
or local neural models with fast fallbacks for offline testing.
"""

import os
import math
import logging
import hashlib
from typing import Optional, List
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_CACHE = None


class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""
    pass


def get_embedding_model():
    """
    Lazy loader for SentenceTransformer embedding model with memory caching.
    """
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is not None:
        return _EMBEDDING_MODEL_CACHE

    model_name = getattr(settings, 'RAG_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer embedding model: '{model_name}'...")
        _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
        return _EMBEDDING_MODEL_CACHE
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer ({model_name}): {str(e)}. Using fallback embedder.")
        return None


def generate_fallback_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Deterministic mathematical hashing vectorizer for testing or offline environments
    when pretrained models cannot be fetched.
    """
    if not text:
        return [0.0] * dim

    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)

    for word in words:
        # Hash word into vector components
        h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def generate_openai_embeddings(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Generate embeddings using OpenAI API (text-embedding-3-small) for serverless operation.
    """
    api_key = getattr(settings, 'OPENAI_API_KEY', '') or os.getenv('OPENAI_API_KEY', '')
    if not api_key or 'your_' in api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.warning(f"OpenAI embedding generation failed ({str(e)}). Using local/fallback vectorizer.")
        return None


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate dense vector embeddings for a list of strings.
    1. Tries OpenAI Embeddings API (fast, cloud, zero serverless bundle size).
    2. Tries local SentenceTransformer if installed.
    3. Falls back to deterministic mathematical hashing vectors.
    """
    if not texts:
        return []

    # Filter/clean strings
    cleaned_texts = [str(t or '').strip() for t in texts]

    # 1. Cloud OpenAI API
    cloud_embeddings = generate_openai_embeddings(cleaned_texts)
    if cloud_embeddings is not None and len(cloud_embeddings) == len(cleaned_texts):
        return cloud_embeddings

    # 2. Local SentenceTransformer if available
    model = get_embedding_model()
    if model is not None:
        try:
            embeddings = model.encode(cleaned_texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.warning(f"SentenceTransformer encoding failed: {str(e)}. Falling back to deterministic vectors.")

    # 3. Fallback mathematical vectorizer
    return [generate_fallback_embedding(t) for t in cleaned_texts]


def generate_embedding(text: str) -> list[float]:
    """
    Generate dense vector embedding for a single string.
    """
    res = generate_embeddings([text])
    if res:
        return res[0]
    return [0.0] * 384

