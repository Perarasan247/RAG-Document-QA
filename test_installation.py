"""
Test script to verify installation
"""

import sys
import importlib


def test_imports():
    """Test if all required packages are installed"""

    packages = [
        "streamlit",
        "fastapi",
        "uvicorn",
        "fitz",  # PyMuPDF
        "pandas",
        "ebooklib",
        "bs4",  # beautifulsoup4
        "langchain",
        "sentence_transformers",
        "faiss",
        "transformers",
        "torch",
    ]

    print("🧪 Testing package imports...")
    print("=" * 50)

    failed = []

    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package:30} - OK")
        except ImportError as e:
            print(f"❌ {package:30} - FAILED")
            failed.append(package)

    print("=" * 50)

    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   • {pkg}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True


def test_directories():
    """Test if required directories exist"""
    import os

    print("\n🗂️  Testing directory structure...")
    print("=" * 50)

    dirs = [
        "data/raw",
        "data/processed",
        "vectorstore",
        "src/loaders",
        "src/chunking",
        "src/embeddings",
        "src/retriever",
    ]

    all_exist = True

    for directory in dirs:
        if os.path.exists(directory):
            print(f"✅ {directory:30} - EXISTS")
        else:
            print(f"❌ {directory:30} - MISSING")
            all_exist = False

    print("=" * 50)

    if not all_exist:
        print("\n❌ Some directories are missing")
        print("💡 Run: mkdir -p data/raw data/processed vectorstore")
        return False
    else:
        print("\n✅ All directories exist!")
        return True


def test_gpu():
    """Test GPU availability"""
    import torch

    print("\n🖥️  Testing GPU availability...")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(
            f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("⚠️  No GPU detected - will use CPU (slower)")

    print("=" * 50)
    return True


def test_models():
    """Test if models can be loaded"""
    print("\n🤖 Testing model loading...")
    print("=" * 50)
    print("⏳ This may take a while on first run (downloading models)...")

    try:
        from sentence_transformers import SentenceTransformer

        print("\n📥 Loading embedding model...")
        model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("✅ Embedding model loaded successfully!")

        # Test embedding
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        print(f"   Embedding shape: {embedding.shape}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("🔍 Document Q&A System - Installation Test")
    print("=" * 50 + "\n")

    results = []

    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Directory Structure", test_directories()))
    results.append(("GPU Availability", test_gpu()))

    # Ask if user wants to test model loading
    print("\n" + "=" * 50)
    response = input("Test model loading? This will download ~400MB (y/n): ").lower()

    if response == "y":
        results.append(("Model Loading", test_models()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")

    print("=" * 50)

    if all(r for _, r in results):
        print("\n🎉 All tests passed! System is ready to use.")
        print("\n🚀 Start the app with: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")

    print()


if __name__ == "__main__":
    main()
