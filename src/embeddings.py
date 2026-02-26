import logging
from typing import Optional, List
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import streamlit as st
import os
import shutil

PERSIST_DIRECTORY = os.path.join("data", "vectors")
logger = logging.getLogger(__name__)

class EmbeddingsStore:

    session_state= None

    def __init__(self, session_state):
        EmbeddingsStore.session_state = session_state

    @staticmethod
    def create_vector_db(file_upload) -> Chroma:
        """
        Create a vector database from an uploaded PDF file.

        Args:
            file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

        Returns:
            Chroma: A vector store containing the processed document chunks.
        """
        logger.info(f"Creating vector DB from file upload: {file_upload.name}")
        
        temp_dir = os.path.join("data","pdfs")
        if not os.path.exists(temp_dir):
            os.makedirs(os.path.join("data","pdfs"))

        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
            logger.info(f"File saved to temporary path: {path}")
            loader = UnstructuredPDFLoader(path)
            data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.getenv("chunk_size",7500)),
                                                        chunk_overlap=int(os.getenv("chunk_overlap", 100)))
        chunks = text_splitter.split_documents(data)
        logger.info("Document split into chunks")

        # Updated embeddings configuration with persistent storage
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"pdf_{hash(file_upload.name)}"  # Unique collection name per file
        )
        logger.info("Vector DB created with persistent storage")

        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} removed")
        return vector_db

    @staticmethod
    def delete_vector_db(vector_db: Optional[Chroma], collection_names : List= None) -> None:
        """
        Delete the vector database and clear related session state.

        Args:
            vector_db (Optional[Chroma]): The vector database to be deleted.
        """
        logger.info("Deleting vector DB")
        if vector_db is not None:
            try:
                for collection_name in collection_names:
                    vector_db.delete_collection(collection_name)
                
                # Clear session state
                EmbeddingsStore.session_state.pop("pdf_pages", None)
                EmbeddingsStore.session_state.pop("file_upload", None)
                EmbeddingsStore.session_state.pop("vector_db", None)
                
                st.success("Collection and temporary files deleted successfully.")
                logger.info("Vector DB and related session state cleared")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting collection: {str(e)}")
                logger.error(f"Error deleting collection: {e}")
        else:
            st.error("No vector database found to delete.")
            logger.warning("Attempted to delete vector DB, but none was found")
