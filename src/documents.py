
import logging
import tempfile, os
from datetime import datetime
import pdfplumber
import shutil
from typing import List, Any
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = os.path.join("data", "vectors")
logger = logging.getLogger(__name__)


class DocumentProcessor:
    
    # Session state
    session_state= None

    def __init__(self, session_state):
        DocumentProcessor.session_state = session_state
    
    @staticmethod
    def process_and_store_pdf(file_upload, pdf_id: str, is_sample: bool = False):
        """Process single PDF and store in session state."""
        logger.info(f"Processing PDF: {file_upload.name} with ID: {pdf_id}")

        # Create temp directory
        temp_dir = os.path.join("data","pdfs")
        if not os.path.exists(temp_dir):
            os.makedirs(os.path.join("data","pdfs"))
        path = os.path.join(temp_dir, file_upload.name)

        # Save file
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {path}")

        # Load and chunk
        loader = UnstructuredPDFLoader(path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.getenv("chunk_size",7000)),
                                                        chunk_overlap=int(os.getenv("chunk_overlap", 100)))
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Add metadata to EACH chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "pdf_id": pdf_id,
                "pdf_name": file_upload.name,
                "chunk_index": i,
                "source_file": file_upload.name
            })

        # Create vector DB with unique collection
        collection_name = f"pdf_{abs(hash(file_upload.name + pdf_id))}"
        logger.info(f"Creating vector DB with collection name: {collection_name}")

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=collection_name
        )
        logger.info("Vector DB created with persistent storage")

        # Extract PDF pages
        with pdfplumber.open(file_upload if not is_sample else path) as pdf:
            pdf_pages = [page.to_image().original for page in pdf.pages]
        logger.info(f"Extracted {len(pdf_pages)} pages from PDF")

        # Store in session state
        DocumentProcessor.session_state["pdfs"][pdf_id] = {
            "name": file_upload.name,
            "vector_db": vector_db,
            "pages": pdf_pages,
            "file_upload": file_upload,
            "collection_name": collection_name,
            "upload_timestamp": datetime.now(),
            "doc_count": len(chunks),
            "is_sample": is_sample
        }
        #st.session_state["active_pdfs"].append(pdf_id)
        DocumentProcessor.session_state["active_pdfs"].insert(0, pdf_id)
        logger.info(f"PDF stored in session state with {len(chunks)} chunks")

        # Cleanup
        # shutil.rmtree(temp_dir)
        # logger.info(f"Temporary directory {temp_dir} removed")

    @staticmethod
    @st.cache_data
    def extract_all_pages_as_images(file_upload) -> List[Any]:
        """
        Extract all pages from a PDF file as images.

        Args:
            file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

        Returns:
            List[Any]: A list of image objects representing each page of the PDF.
        """
        logger.info(f"Extracting all pages as images from file: {file_upload.name}")
        pdf_pages = []
        with pdfplumber.open(file_upload) as pdf:
            pdf_pages = [page.to_image().original for page in pdf.pages]
        logger.info("PDF pages extracted as images")
        return pdf_pages


    @staticmethod
    def delete_pdf(pdf_id: str):
        """Delete single PDF and its collection."""
        if pdf_id in DocumentProcessor.session_state["pdfs"]:
            pdf_data = st.session_state["pdfs"][pdf_id]
            logger.info(f"Deleting PDF: {pdf_data['name']} (ID: {pdf_id})")

            # Delete vector collection
            try:
                pdf_data["vector_db"].delete_collection()
                logger.info(f"Deleted collection: {pdf_data['collection_name']}")
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")

            # Remove from state
            del DocumentProcessor.session_state["pdfs"][pdf_id]
            DocumentProcessor.session_state["active_pdfs"].remove(pdf_id)

            st.success(f"Deleted {pdf_data['name']}")

    @staticmethod
    def delete_all_pdfs():
        """Delete all PDFs."""
        logger.info("Deleting all PDFs")
        for pdf_id in list(DocumentProcessor.session_state["pdfs"].keys()):
            DocumentProcessor.delete_pdf(pdf_id)
        DocumentProcessor.session_state["pdfs"] = {}
        DocumentProcessor.session_state["active_pdfs"] = []
