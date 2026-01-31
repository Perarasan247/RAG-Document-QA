"""
Vector store using FAISS for similarity search - CPU Optimized
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional


class VectorStore:
    """FAISS-based vector store with metadata - CPU Optimized"""

    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index - CPU Optimized

        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        # Use IndexFlatIP (Inner Product) for CPU - simple and fast
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        self.doc_id_to_indices = {}  # Map document name to chunk indices

    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add documents to the vector store

        Args:
            embeddings: Numpy array of embeddings (n_docs, dimension)
            metadata: List of metadata dictionaries
        """
        # Add embeddings to FAISS
        self.index.add(embeddings.astype("float32"))

        # Store metadata
        start_idx = len(self.metadata)
        self.metadata.extend(metadata)

        # Build document index mapping
        for i, meta in enumerate(metadata):
            doc_name = meta["source"]
            if doc_name not in self.doc_id_to_indices:
                self.doc_id_to_indices[doc_name] = []
            self.doc_id_to_indices[doc_name].append(start_idx + i)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        doc_filter: Optional[str] = None,
        threshold: float = 0.3,
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            doc_filter: Optional document name to filter results
            threshold: Minimum similarity score

        Returns:
            List of (metadata, score) tuples
        """
        # Reshape query for FAISS
        query_vector = query_embedding.reshape(1, -1).astype("float32")

        if doc_filter and doc_filter in self.doc_id_to_indices:
            # Filter search to specific document
            relevant_indices = self.doc_id_to_indices[doc_filter]

            # Create a subset index
            subset_embeddings = []
            subset_metadata = []
            for idx in relevant_indices:
                vector = self.index.reconstruct(int(idx))
                subset_embeddings.append(vector)
                subset_metadata.append(self.metadata[idx])

            if not subset_embeddings:
                return []

            # Search in subset
            subset_embeddings = np.array(subset_embeddings).astype("float32")
            scores = np.dot(subset_embeddings, query_vector.T).flatten()
            top_k_indices = np.argsort(scores)[::-1][:k]

            results = []
            for idx in top_k_indices:
                score = float(scores[idx])
                if score >= threshold:
                    results.append((subset_metadata[idx], score))
        else:
            # Search entire index
            scores, indices = self.index.search(query_vector, k)
            scores = scores[0]
            indices = indices[0]

            results = []
            for idx, score in zip(indices, scores):
                if score >= threshold:
                    results.append((self.metadata[int(idx)], float(score)))

        return results

    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "doc_id_to_indices": self.doc_id_to_indices,
                },
                f,
            )

        print(f"Vector store saved to {index_path}")

    def load(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata"""
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)

            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.doc_id_to_indices = data["doc_id_to_indices"]

            print(f"Vector store loaded from {index_path}")
            return True
        return False

    def get_document_list(self) -> List[str]:
        """Get list of all documents in the store"""
        return list(self.doc_id_to_indices.keys())

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(self.doc_id_to_indices),
            "documents": list(self.doc_id_to_indices.keys()),
        }
