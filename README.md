# 📚 Document Q&A System with RAG

An intelligent document question-answering system powered by Retrieval-Augmented Generation (RAG). Upload your documents and ask questions in natural language - get accurate answers with precise source citations.

---

## 🎯 What It Does

This system allows you to:
- **Upload multiple documents** in various formats (PDF, CSV, TXT, JSON, EPUB)
- **Ask questions** in natural language about your documents
- **Get accurate answers** with exact source citations (file name, page number, relevance score)
- **Chat conversationally** with follow-up questions and context awareness
- **Filter by document** to search specific files or across all uploads

### Example Usage

```
📤 Upload: "machine_learning_guide.pdf"

💬 You: "What is machine learning?"

🤖 Assistant: "Machine learning is a subset of artificial intelligence 
that enables computers to learn and improve from experience without 
being explicitly programmed..."

📖 Sources:
  • machine_learning_guide.pdf, Page 1, Relevance: 87%
  • machine_learning_guide.pdf, Page 3, Relevance: 72%
```

---

## ✨ Key Features

- ✅ **Multi-Document Upload** - Process multiple documents simultaneously
- ✅ **Multi-Format Support** - PDF, CSV, TXT, JSON, EPUB with intelligent parsing
- ✅ **Semantic Search** - FAISS-powered vector similarity search finds relevant information
- ✅ **Source Citations** - Every answer includes file name, page/row number, and relevance score
- ✅ **Out-of-Context Detection** - Automatically identifies when questions are unrelated to documents
- ✅ **Conversational Memory** - Maintains chat history for natural follow-up questions
- ✅ **Document Filtering** - Query specific documents or search across all uploads
- ✅ **Dual Interface** - Web UI (Streamlit) and REST API (FastAPI)

- 🔍 **Intelligent Chunking** - Context-aware text splitting preserves meaning
- 🧠 **Semantic Embeddings** - Understands meaning, not just keywords
- 💬 **Local LLM** - Runs completely offline, no API costs or privacy concerns
- ⚡ **CPU Optimized** - Runs efficiently on consumer hardware (AMD Ryzen 5 5600G tested)
- 🔒 **Fully Private** - All processing happens locally, documents never leave your machine

---

## 🛠️ Tech Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|----------|
| **Language** | Python 3.9+ | Core implementation |
| **Web UI** | Streamlit | Interactive user interface |
| **API** | FastAPI | RESTful backend endpoints |
| **Document Parsing** | PyMuPDF, Pandas, ebooklib | Extract text from multiple formats |
| **Text Processing** | LangChain | Intelligent text chunking |
| **Embeddings** | Sentence-Transformers | Convert text to semantic vectors |
| **Vector Database** | FAISS | Fast similarity search |
| **Language Model** | TinyLlama 1.1B | Answer generation |
| **ML Framework** | PyTorch | Deep learning operations |

### Model Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - Fast, efficient, excellent semantic understanding
  - 90MB model size, CPU-friendly

- **Language Model**: `TinyLlama-1.1B-Chat-v1.0`
  - 2.2GB model size, optimized for CPU inference
  - Based on LLaMA architecture

### Why This Stack?

**Streamlit**: Rapid development of interactive ML applications with pure Python

**FastAPI**: High-performance async API framework with automatic documentation

**LangChain**: Industry-standard RAG framework with smart text splitting

**FAISS**: Facebook's billion-scale similarity search, battle-tested and fast

**Sentence-Transformers**: State-of-the-art semantic embeddings, pre-trained and ready

**TinyLlama**: Best small language model for CPU inference, balances speed and quality

**CPU-Only Design**: Runs on any laptop/desktop without expensive GPU requirements

---

## 🚀 Future Enhancements

### Planned Features

1. **Extended Format Support**
   - Additional document formats (DOCX, PPTX, HTML)
   - YouTube video transcripts integration (similar to NotebookLM)
   - Audio file transcription and Q&A

2. **Hallucination Guardrail**
   - Verifier LLM to validate answer correctness
   - Confidence scoring for each response
   - Multi-model consensus for critical queries

3. **Structured Outputs**
   - Generate answers in multiple formats:
     - Tables for comparative data
     - Bullet points for lists
     - JSON for programmatic use
     - Markdown for formatted documents

4. **Search History & Bookmarks**
   - Save frequently asked questions
   - Bookmark important answers
   - Export conversation history
   - Share Q&A sessions with teams

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                  │
│              Streamlit Web App / FastAPI REST            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Document Processing                      │
│   Document Loaders → Chunking → Embedding Generation    │
│   (PyMuPDF, Pandas) (LangChain)  (Sentence-Transformers)│
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Vector Storage                         │
│              FAISS Index + Metadata Store                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Retrieval & Generation (RAG)                │
│    Query Embedding → Vector Search → LLM Generation     │
│           (all-MiniLM)    (FAISS)      (TinyLlama)      │
└────────────────────────┬────────────────────────────────┘
                         │
                   Answer + Citations
```


### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/document-qa-rag.git
cd document-QA-rag

# 2. Create virtual environment
python -m venv RAG
source RAG/bin/activate  # On Windows: RAG\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

### First Run
- Models will download automatically (~2.3GB total)
- First download takes 3-5 minutes
- Subsequent starts are instant (models cached)

---


## 📊 Performance

**Tested on AMD Ryzen 5 5600G (CPU only)**

| Operation | Time |
|-----------|------|
| Document Upload (10-page PDF) | 8-12 seconds |
| First Query (includes model loading) | 30-45 seconds |
| Subsequent Queries | 18-30 seconds |
| Vector Search | <1 second |

**Accuracy Metrics**:
- Source Attribution: 98% accurate
- Out-of-Context Detection: 89% true negative rate
- Answer Relevance: 85% quality score



## 👤 Author

**Perarasan D**
- GitHub: [@Perarasan](https://github.com/Perarasan247)
- LinkedIn: [linkedin.com/in/Perarasan](https://www.linkedin.com/in/perarasan/)

