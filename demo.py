"""
Demo script to test the Document Q&A system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_qa_system import DocumentQASystem


def run_demo():
    """Run a simple demo of the system"""

    print("\n" + "=" * 60)
    print("📚 Document Q&A System - Demo")
    print("=" * 60 + "\n")

    # Initialize system
    print("🔧 Initializing system (CPU Mode - Optimized for Ryzen 5 5600G)...")
    qa_system = DocumentQASystem(
        data_dir="data/raw",
        vector_store_dir="vectorstore",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    print("✅ System initialized!\n")

    # Check if sample document exists
    sample_doc = "sample_document.txt"

    if Path(sample_doc).exists():
        print(f"📄 Found sample document: {sample_doc}")

        # Upload document
        print("📤 Processing document...")
        result = qa_system.upload_document(sample_doc)

        if result["success"]:
            print(f"✅ {result['message']}")
            print(f"   Chunks created: {result['chunks']}\n")
        else:
            print(f"❌ {result['message']}\n")
            return
    else:
        print("⚠️  Sample document not found. Please upload a document first.\n")
        return

    # Show documents
    print("=" * 60)
    docs = qa_system.get_documents()
    print(f"📚 Uploaded documents ({len(docs)}):")
    for doc in docs:
        print(f"   • {doc}")

    # Show stats
    stats = qa_system.get_stats()
    print(f"\n📊 Total chunks: {stats['total_chunks']}")

    # Demo queries
    print("\n" + "=" * 60)
    print("💬 Demo Queries")
    print("=" * 60 + "\n")

    queries = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "Explain overfitting and how to prevent it",
        "What is deep learning?",
        "What is the capital of France?",  # Out of context
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Query {i}: {query}")
        print("─" * 60)

        result = qa_system.query(query, top_k=3)

        # Display answer
        print(f"\n📝 Answer:")
        print(f"{result['answer']}\n")

        # Display sources
        if result["sources"]:
            print("📖 Sources:")
            for j, source in enumerate(result["sources"], 1):
                print(
                    f"   {j}. {source['source']} (Page {source.get('page', 'N/A')}) - Relevance: {source['score']:.2%}"
                )

        if result["out_of_context"]:
            print("\n⚠️  This query is out of context!")

    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)

    print("\n🚀 Next steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Run the API server: python api.py")
    print("   3. Upload your own documents and ask questions!\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed all dependencies: pip install -r requirements.txt")
        print("   2. Have enough disk space for model downloads (~6GB)")
        print("   3. Have internet connection for first run")
