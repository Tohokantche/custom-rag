"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

import streamlit as st
import logging
import os
import ollama
from pathlib import Path
from langchain_core.messages import  AIMessage, SystemMessage
from src.utils import render_config, extract_model_names, generate_pdf_id, LoadReRanker, DownloadModels
from src.documents import DocumentProcessor
from src.embeddings import EmbeddingsStore, PERSIST_DIRECTORY
from src.rag import RagLogic

# Suppress torch warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Plug-and-play Ollama RAG Streamlit UI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Plug-and-play Ollama RAG", divider="gray", anchor=False)

    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    download_attempt = 0
    model_threads = None

    # Embedding model, Chat_model, Re_ranker model
    model_names =["nomic-embed-text:latest", "qwen3:1.7b", "zeroentropy/zerank-1-small"]

    # In case the user failed to manually download the models we did it for him/her
    while (len(available_models) <2) and (download_attempt < 2) :
        with st.status("Downloading embeddings and chat models ..."):
            model_threads = [DownloadModels(model_name) for model_name in model_names[:-1]].append(LoadReRanker(model_names[-1]))
            for thread in model_threads:
                thread.start()
            for thread in model_threads:
                thread.join()
            for i, thread in enumerate(model_threads):
                if thread.return_value:
                    logger.info(f"Sucessfully download models: {thread.model_name}")
                
            available_models = extract_model_names(models_info) 
            download_attempt +=1

    # Create layout
    col1, col2 = st.columns([1, 1])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["lang_messages"] = [
                    SystemMessage(content="You are a helpful assistant.")
                ]
    if "pdfs" not in st.session_state:
        st.session_state["pdfs"] = {}
    if "active_pdfs" not in st.session_state:
        st.session_state["active_pdfs"] = []
    if "new_pdfs" not in st.session_state:
        st.session_state["new_pdfs"] = False
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False
    if "use_re_ranker" not in st.session_state:
        st.session_state["use_re_ranker"] = False
    if "re_ranker" not in st.session_state:
        st.session_state["re_ranker"] = None

    # Instantiate classes
    doc_processor = DocumentProcessor(st.session_state)
    emb_store = EmbeddingsStore(st.session_state)

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓",
            available_models,
            key="model_select"
        )

    # PDF Management UI in Sidebar
    with st.sidebar:
        st.divider()
        selection = render_config()
        if selection:
            os.environ["chunk_size"]= selection["chunk_size"]
            os.environ["chunk_overlap"]= selection["chunk_overlap"]
            if selection["re_ranker"]:
                st.session_state["use_re_ranker"] = True
                with st.spinner("Loading re-ranker model ..."):
                    if model_threads:
                        st.session_state["re_ranker"] = model_threads[-1].return_value if model_threads[-1].return_value else None
                    if not st.session_state["re_ranker"]:
                        load_re_ranker =LoadReRanker(model_names[-1])
                        load_re_ranker.start()
                        load_re_ranker.join()
                        st.session_state["re_ranker"] = load_re_ranker.return_value
                    logger.info(f"Sucessfully load the re-ranker model")

        st.divider()
        st.subheader("📚 Loaded PDFs")

        if st.session_state.get("pdfs"):
            total_pdfs = len(st.session_state["pdfs"])
            total_chunks = sum(pdf["doc_count"] for pdf in st.session_state["pdfs"].values())

            st.metric("Total PDFs", total_pdfs)
            st.metric("Total Chunks", total_chunks)
            st.divider()

            # List PDFs
            for pdf_id in st.session_state["active_pdfs"]:
                pdf_data = st.session_state["pdfs"][pdf_id]

                with st.expander(f"📄 {pdf_data['name']}", expanded=False):
                    st.caption(f"Chunks: {pdf_data['doc_count']}")
                    st.caption(f"Pages: {len(pdf_data['pages'])}")

                    if st.button("🗑️ Delete", key=f"delete_{pdf_id}"):
                        doc_processor.delete_pdf(pdf_id)
                        st.session_state["new_pdfs"] = False
                        st.rerun()

            st.divider()
            if st.button("🗑️ Delete All PDFs"):
                doc_processor.delete_all_pdfs()
                st.rerun()
        else:
            st.info("No PDFs loaded yet.")
        # Put app configuration here (Chunck_size, chunk_overlap, and )

    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Use sample PDF (World Economic Forum 2026 report)", 
        key="sample_checkbox"
    )
    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample

    if use_sample:
        # Use the sample PDF
        sample_pdf_path = Path("data/pdfs/sample/WEF_Global_Risks_Report_2026.pdf")
        # Reset the context
        st.session_state["lang_messages"] = [SystemMessage(content="You are a helpful assistant.")]
        if sample_pdf_path.exists():
            # Check if already loaded
            sample_id = "sample_pdf"
            if sample_id not in st.session_state.get("pdfs", {}):
                with st.spinner("Loading sample PDF..."):
                    # Create a file-like object
                    with open(sample_pdf_path, "rb") as f:
                        file_bytes = f.read()

                    # Create UploadedFile-like object
                    class SampleFile:
                        def __init__(self, path, content):
                            self.name = path.name
                            self._content = content

                        def getvalue(self):
                            return self._content

                    sample_file = SampleFile(sample_pdf_path, file_bytes)
                    doc_processor.process_and_store_pdf(sample_file, sample_id, is_sample=True)
        else:
            st.error("Sample PDF file not found in the current directory.")
    else:
        # Regular file upload with multi-file support
        #file_uploads= None
        file_uploads = col1.file_uploader(
            "Upload PDF files ↓",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        # Reset the context
        st.session_state["lang_messages"] = [SystemMessage(content="You are a helpful assistant.")]
        if file_uploads:
            for file_upload in file_uploads:
                pdf_id = generate_pdf_id(file_upload)

                # Skip if already processed
                if pdf_id not in st.session_state.get("pdfs", {}):
                    with st.spinner(f"Processing {file_upload.name}..."):
                        doc_processor.process_and_store_pdf(file_upload, pdf_id)
                        st.session_state["new_pdfs"] = True
                else:
                    st.session_state["new_pdfs"] = False

    # Stacked PDF Viewer
    if st.session_state.get("pdfs") and st.session_state["active_pdfs"]:
        
        zoom_level = col1.slider(
            "Zoom Level",
            min_value=100,
            max_value=1000,
            value=700,
            step=50,
            key="zoom_slider"
        )      
        with col1:
            
            with st.container(height=410, border=True):
                for pdf_id in st.session_state["active_pdfs"]:
                    if pdf_id not in st.session_state["pdfs"]:
                        continue

                    pdf_data = st.session_state["pdfs"][pdf_id]

                    # PDF header with metadata
                    st.markdown(f"### 📄 {pdf_data['name']}")
                    st.caption(
                        f"Uploaded: {pdf_data['upload_timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                        f"Chunks: {pdf_data['doc_count']} | "
                        f"Pages: {len(pdf_data['pages'])}"
                    )

                    # Quick remove button
                    if st.button("🗑️ Remove", key=f"remove_{pdf_id}"):
                        doc_processor.delete_pdf(pdf_id)
                        st.session_state["new_pdfs"] = False
                        st.rerun()

                    st.divider()

                    # Display all pages
                    for page_idx, page_image in enumerate(pdf_data['pages']):
                        st.caption(f"Page {page_idx + 1}")
                        st.image(page_image, width=zoom_level)

                    # Spacing between PDFs
                    st.markdown("---")
                if st.session_state["new_pdfs"]:
                    st.session_state["new_pdfs"] = False
                    st.rerun() 
    else:
        col1.info("Upload PDF files to view them here.")

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        emb_store.delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

                # Show sources if available
                if message["role"] == "assistant" and "sources" in message:
                    st.divider()
                    st.caption("📚 Sources:")

                    sources_by_pdf = {}
                    for src in message["sources"]:
                        pdf_name = src.get("pdf_name", "Unknown")
                        if pdf_name not in sources_by_pdf:
                            sources_by_pdf[pdf_name] = 0
                        sources_by_pdf[pdf_name] += 1

                    for pdf_name, count in sources_by_pdf.items():
                        st.markdown(f"- **{pdf_name}** ({count} chunks)")

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state.get("pdfs"):
                            response, sources = RagLogic.process_question_multi_pdf(
                                prompt,
                                st.session_state["pdfs"],
                                selected_model,
                                st.session_state["lang_messages"],
                                {"re_ranker": st.session_state["re_ranker"] if st.session_state["use_re_ranker"] else None,
                                 "h_search": selection['h_search'] 
                                 }
                            )
                            st.markdown(response)

                            # Display sources
                            if sources:
                                st.divider()
                                st.caption("📚 Sources:")

                                # Group by PDF
                                sources_by_pdf = {}
                                for src in sources:
                                    pdf_name = src.get("pdf_name", "Unknown")
                                    if pdf_name not in sources_by_pdf:
                                        sources_by_pdf[pdf_name] = 0
                                    sources_by_pdf[pdf_name] += 1

                                for pdf_name, count in sources_by_pdf.items():
                                    st.markdown(f"- **{pdf_name}** ({count} chunks)")
                        else:
                            st.warning("Please upload PDF files first.")
                            response = None
                            sources = None

                # Add assistant response to chat history with sources
                if response:
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    st.session_state["lang_messages"].append(AIMessage(content=response)) 

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"An unexpected error occurs : {e}")
        else:
            if not st.session_state.get("pdfs"):
                st.warning("Upload PDF files or use the sample PDF to begin chat ...")


if __name__ == "__main__":
    main()
