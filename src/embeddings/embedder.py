"""
Embeddings generation using sentence-transformers - CPU Optimized
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for text chunks - CPU Optimized"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model - CPU Optimized

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, 384 dims)
        """
        print(f"Loading embedding model: {model_name}")
        print("⚙️  CPU Mode - This may take a moment...")

        # Load model with CPU-friendly settings
        self.model = SentenceTransformer(model_name, device="cpu")

        print("✅ Embedding model loaded successfully")

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Generate embeddings for a list of documents

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding (smaller for CPU)

        Returns:
            Numpy array of embeddings
        """
        print(f"  Encoding {len(texts)} chunks in batches of {batch_size}...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,  # Smaller batches for CPU
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query

        Args:
            query: Query string

        Returns:
            Numpy array embedding
        """
        embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding[0]
