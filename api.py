"""
FastAPI backend for Document Q&A System
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from src.document_qa_system import DocumentQASystem

# Initialize FastAPI
app = FastAPI(
    title="Document Q&A API",
    description="RAG-based Document Question Answering System",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QA system - CPU Optimized for Ryzen 5 5600G
qa_system = DocumentQASystem(
    data_dir="data/raw",
    vector_store_dir="vectorstore",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)


# Request models
class QueryRequest(BaseModel):
    question: str
    doc_filter: Optional[str] = None
    top_k: int = 5
    use_conversation: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    out_of_context: bool


# API Endpoints
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Document Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "documents": "/documents",
            "stats": "/stats",
            "reset": "/reset",
            "clear": "/clear",
        },
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document

    Args:
        file: Document file (PDF, TXT, CSV, JSON, EPUB)

    Returns:
        Processing status
    """
    # Validate file type
    allowed_extensions = [".pdf", ".txt", ".csv", ".json", ".epub"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save temporarily using proper temp directory (cross-platform)
    import tempfile

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process document
        result = qa_system.upload_document(temp_path, file.filename)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if result["success"]:
            return {
                "status": "success",
                "message": result["message"],
                "filename": file.filename,
                "chunks": result["chunks"],
            }
        else:
            raise HTTPException(status_code=400, detail=result["message"])

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query the document Q&A system

    Args:
        request: Query request with question and options

    Returns:
        Answer with sources
    """
    try:
        result = qa_system.query(
            question=request.question,
            doc_filter=request.doc_filter,
            top_k=request.top_k,
            use_conversation=request.use_conversation,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    """
    Get list of uploaded documents

    Returns:
        List of document names
    """
    try:
        docs = qa_system.get_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """
    Get system statistics

    Returns:
        Statistics about documents and chunks
    """
    try:
        stats = qa_system.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset_conversation():
    """
    Reset conversation history

    Returns:
        Success message
    """
    try:
        qa_system.reset_conversation()
        return {"status": "success", "message": "Conversation history reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
def clear_all():
    """
    Clear all documents and vector store

    Returns:
        Success message
    """
    try:
        qa_system.clear_all()
        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
