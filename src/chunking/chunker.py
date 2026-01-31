"""
Text chunking with metadata preservation
"""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Chunk documents while preserving source metadata"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, raw_chunks: List[Dict]) -> List[Dict]:
        """
        Split documents into smaller chunks while preserving metadata

        Args:
            raw_chunks: List of document chunks with metadata

        Returns:
            List of smaller chunks with preserved metadata
        """
        chunked_docs = []

        for doc in raw_chunks:
            text = doc["text"]

            # Skip empty text
            if not text.strip():
                continue

            # Split the text
            splits = self.text_splitter.split_text(text)

            # Add metadata to each split
            for i, split in enumerate(splits):
                chunk = {
                    "text": split,
                    "source": doc["source"],
                    "page": doc.get("page", 1),
                    "doc_type": doc["doc_type"],
                    "file_path": doc["file_path"],
                    "chunk_id": i,
                }

                # Preserve additional metadata
                if "row" in doc:
                    chunk["row"] = doc["row"]
                if "chapter" in doc:
                    chunk["chapter"] = doc["chapter"]

                chunked_docs.append(chunk)

        return chunked_docs
