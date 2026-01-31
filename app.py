"""
Streamlit UI for Document Q&A System
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_qa_system import DocumentQASystem


# Page configuration
st.set_page_config(page_title="Document Q&A with RAG", page_icon="📚", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .out-of-context {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_system():
    """Initialize the Document Q&A system (cached) - CPU Optimized"""
    with st.spinner(
        "🔄 Initializing system (CPU Mode - Optimized for Ryzen 5 5600G)..."
    ):
        return DocumentQASystem(
            data_dir="data/raw",
            vector_store_dir="vectorstore",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )


def main():
    # Header
    st.markdown(
        '<div class="main-header">📚 Document Q&A System with RAG</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Initialize system
    try:
        qa_system = initialize_system()
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.info(
            "💡 Make sure you have installed all requirements: `pip install -r requirements.txt`"
        )
        return

    # Sidebar for document management
    with st.sidebar:
        st.header("📂 Document Management")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "csv", "json", "epub"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, CSV, JSON, EPUB",
        )

        if uploaded_files:
            if st.button("📤 Process Uploaded Files"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Create temp directory if it doesn't exist
                import tempfile

                temp_dir = tempfile.gettempdir()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    # Save temporarily using proper temp directory
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Process
                    result = qa_system.upload_document(temp_path, uploaded_file.name)

                    if result["success"]:
                        st.success(
                            f"✅ {uploaded_file.name}: {result['chunks']} chunks"
                        )
                    else:
                        st.error(f"❌ {uploaded_file.name}: {result['message']}")

                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("✅ All files processed!")
                st.rerun()

        st.markdown("---")

        # Current documents
        st.subheader("📄 Uploaded Documents")
        docs = qa_system.get_documents()

        if docs:
            for doc in docs:
                st.text(f"• {doc}")

            # Statistics
            stats = qa_system.get_stats()
            st.info(
                f"📊 Total: {stats['total_documents']} documents, {stats['total_chunks']} chunks"
            )
        else:
            st.info("No documents uploaded yet")

        st.markdown("---")

        # Clear options
        st.subheader("🗑️ Clear Data")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset Chat"):
                qa_system.reset_conversation()
                if "messages" in st.session_state:
                    st.session_state.messages = []
                st.success("Chat reset!")
                st.rerun()

        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                if st.button("⚠️ Confirm?", type="primary"):
                    qa_system.clear_all()
                    if "messages" in st.session_state:
                        st.session_state.messages = []
                    st.success("All data cleared!")
                    st.rerun()

    # Main chat interface
    st.header("💬 Ask Questions")

    # Query options
    col1, col2 = st.columns([3, 1])

    with col1:
        doc_filter = st.selectbox(
            "Filter by document (optional)",
            ["All documents"] + qa_system.get_documents(),
            help="Search only in specific document",
        )

    with col2:
        use_conversation = st.checkbox(
            "Use conversation history", value=True, help="Enable follow-up questions"
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(
                            f"""
                        <div class="source-box">
                            <b>Source {i}:</b> {source["source"]}<br>
                            <b>Type:</b> {source["doc_type"]}<br>
                            <b>Page:</b> {source.get("page", "N/A")}<br>
                            <b>Relevance:</b> {source["score"]:.2%}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are uploaded
        if not qa_system.get_documents():
            st.warning("⚠️ Please upload documents first!")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Prepare filter
                filter_doc = None if doc_filter == "All documents" else doc_filter

                # Query system
                result = qa_system.query(
                    prompt, doc_filter=filter_doc, use_conversation=use_conversation
                )

                # Display answer
                if result["out_of_context"]:
                    st.markdown(
                        f"""
                    <div class="out-of-context">
                        ⚠️ <b>Out of Context</b><br>
                        {result["answer"]}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(result["answer"])

                # Display sources
                if result["sources"]:
                    with st.expander("📖 View Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(
                                f"""
                            <div class="source-box">
                                <b>Source {i}:</b> {source["source"]}<br>
                                <b>Type:</b> {source["doc_type"]}<br>
                                <b>Page:</b> {source.get("page", "N/A")}<br>
                                <b>Relevance:</b> {source["score"]:.2%}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                # Add assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )


if __name__ == "__main__":
    main()
