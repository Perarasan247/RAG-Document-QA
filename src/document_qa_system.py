"""
Main document processing orchestrator - CPU Optimized
"""

import os
from typing import List, Optional, Dict
from src.loaders.document_loader import DocumentLoader
from src.chunking.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingGenerator
from src.retriever.vector_store import VectorStore
from src.rag import RAGPipeline
import shutil


class DocumentQASystem:
    """Main system orchestrator - Optimized for Ryzen 5 5600G CPU"""

    def __init__(
        self,
        data_dir: str = "data/raw",
        vector_store_dir: str = "vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        """
        Initialize the Document Q&A system - CPU Optimized

        Args:
            data_dir: Directory for uploaded documents
            vector_store_dir: Directory for vector store
            embedding_model: Embedding model (default: smaller model for CPU)
            llm_model: LLM model (default: TinyLlama for CPU)
        """
        print("🚀 Initializing Document Q&A System (CPU Mode)")
        print("⚙️  Optimized for Ryzen 5 5600G")

        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)

        # Initialize components
        print("\n📚 Loading document loader...")
        self.loader = DocumentLoader()

        print("✂️  Loading chunker...")
        self.chunker = DocumentChunker(
            chunk_size=800, chunk_overlap=150
        )  # Smaller chunks for CPU

        print("🔢 Loading embedding model (this may take a moment)...")
        self.embedder = EmbeddingGenerator(model_name=embedding_model)

        print("💾 Initializing vector store...")
        self.vector_store = VectorStore(
            dimension=384
        )  # all-MiniLM-L6-v2 uses 384 dimensions

        # Try to load existing vector store
        self._load_vector_store()

        # Initialize RAG pipeline (lazy loading)
        self.rag_pipeline = None
        self.llm_model = llm_model

        print("✅ System initialization complete!\n")

    def _load_vector_store(self):
        """Load existing vector store if available"""
        index_path = os.path.join(self.vector_store_dir, "index.faiss")
        metadata_path = os.path.join(self.vector_store_dir, "metadata.pkl")

        if self.vector_store.load(index_path, metadata_path):
            print(
                f"📂 Loaded {self.vector_store.index.ntotal} chunks from vector store"
            )

    def _save_vector_store(self):
        """Save vector store"""
        index_path = os.path.join(self.vector_store_dir, "index.faiss")
        metadata_path = os.path.join(self.vector_store_dir, "metadata.pkl")
        self.vector_store.save(index_path, metadata_path)

    def _init_rag_pipeline(self):
        """Initialize RAG pipeline (lazy loading)"""
        if self.rag_pipeline is None:
            print("\n🤖 Loading LLM (this will take a few minutes on first run)...")
            self.rag_pipeline = RAGPipeline(model_name=self.llm_model, device="cpu")

    def upload_document(self, file_path: str, file_name: Optional[str] = None) -> Dict:
        """
        Upload and process a document

        Args:
            file_path: Path to the document
            file_name: Optional custom filename

        Returns:
            Processing status
        """
        if not os.path.exists(file_path):
            return {"success": False, "message": "File not found"}

        # Copy to data directory
        if file_name is None:
            file_name = os.path.basename(file_path)

        dest_path = os.path.join(self.data_dir, file_name)

        # Check if already exists
        if os.path.exists(dest_path):
            return {"success": False, "message": "Document already uploaded"}

        shutil.copy(file_path, dest_path)

        # Process document
        try:
            print(f"\n📄 Processing: {file_name}")

            # Load document
            print("  ⏳ Loading document...")
            raw_chunks = self.loader.load_document(dest_path)

            if not raw_chunks:
                os.remove(dest_path)
                return {
                    "success": False,
                    "message": "Failed to extract content from document",
                }

            # Chunk document
            print(f"  ⏳ Chunking document ({len(raw_chunks)} raw chunks)...")
            chunks = self.chunker.chunk_documents(raw_chunks)

            # Generate embeddings
            print(f"  ⏳ Generating embeddings for {len(chunks)} chunks...")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedder.embed_documents(texts)

            # Add to vector store
            print("  ⏳ Adding to vector store...")
            self.vector_store.add_documents(embeddings, chunks)

            # Save vector store
            self._save_vector_store()

            print(f"  ✅ Successfully processed {file_name}")

            return {
                "success": True,
                "message": f"Successfully processed {file_name}",
                "chunks": len(chunks),
            }

        except Exception as e:
            # Clean up if processing fails
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return {"success": False, "message": f"Error processing document: {str(e)}"}

    def query(
        self,
        question: str,
        doc_filter: Optional[str] = None,
        top_k: int = 4,  # Reduced for CPU
        use_conversation: bool = True,
    ) -> Dict:
        """
        Query the document Q&A system

        Args:
            question: User question
            doc_filter: Optional document name to filter
            top_k: Number of chunks to retrieve (reduced for CPU)
            use_conversation: Use conversation history

        Returns:
            Answer with sources
        """
        # Initialize RAG pipeline if needed
        self._init_rag_pipeline()

        print(f"\n🔍 Processing query: {question[:50]}...")

        # Generate query embedding
        print("  ⏳ Generating query embedding...")
        query_embedding = self.embedder.embed_query(question)

        # Retrieve relevant chunks
        print("  ⏳ Searching documents...")
        retrieved = self.vector_store.search(
            query_embedding,
            k=top_k,
            doc_filter=doc_filter,
            threshold=0.15,  # Lowered threshold for better recall
        )

        if not retrieved:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": [],
                "out_of_context": True,
            }

        # Generate answer using RAG
        print("  ⏳ Generating answer (this may take 10-30 seconds on CPU)...")
        result = self.rag_pipeline.generate_answer(
            question, retrieved, use_history=use_conversation
        )

        print("  ✅ Answer generated!")

        return result

    def get_documents(self) -> List[str]:
        """Get list of uploaded documents"""
        return self.vector_store.get_document_list()

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.vector_store.get_stats()

    def reset_conversation(self):
        """Reset conversation history"""
        if self.rag_pipeline:
            self.rag_pipeline.reset_conversation()

    def clear_all(self):
        """Clear all documents and vector store"""
        # Clear data directory
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
            os.makedirs(self.data_dir)

        # Clear vector store
        if os.path.exists(self.vector_store_dir):
            shutil.rmtree(self.vector_store_dir)
            os.makedirs(self.vector_store_dir)

        # Reinitialize vector store
        self.vector_store = VectorStore(dimension=384)

        # Reset conversation
        if self.rag_pipeline:
            self.rag_pipeline.reset_conversation()

        print("All data cleared")
