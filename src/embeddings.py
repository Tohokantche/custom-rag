import logging
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import streamlit as st
from unstructured.cleaners.core import clean, group_broken_paragraphs
import os, re

# Paddle OCR for ingestion
# os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"

PERSIST_DIRECTORY = os.path.join("data", "vectors")
logger = logging.getLogger(__name__)

class EmbeddingsStore:

    session_state= None
    emb_model_name = None

    def __init__(self, emb_model_name : str, session_state):
      
        EmbeddingsStore.emb_model_name = emb_model_name
        EmbeddingsStore.session_state = session_state

    @staticmethod
    def create_vector_db(file_upload, path, pdf_id: str) -> Chroma:
        """
        Create a vector database from an uploaded PDF file.

        Args:
            file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

        Returns:
            Chroma: A vector store containing the processed document chunks.
        """
        logger.info(f"Creating vector DB from file upload: {file_upload.name}")

        if not path :
            # Create temp directory
            temp_dir = os.path.join("data","pdfs")
            if not os.path.exists(temp_dir):
                os.makedirs(os.path.join("data","pdfs"))
            path = os.path.join(temp_dir, file_upload.name)
            
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(file_upload.getvalue())
                logger.info(f"File saved to temporary path: {path}")

        # Load, clean and create chunk
        loader = UnstructuredPDFLoader(
            path,
            post_processors=[
                    lambda text: group_broken_paragraphs(EmbeddingsStore.remove_page_number_text(clean(text, 
                            extra_whitespace=False, dashes=True, trailing_punctuation= True,lowercase = True)))
                             ]
            )
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.getenv("chunk_size",1500)),
                                                        chunk_overlap=int(os.getenv("chunk_overlap", 80)),
                                                        add_start_index=True)
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
            logger.info(f"Chunck info : {chunk.metadata}")

        # Create vector DB with unique collection
        collection_name = f"pdf_{abs(hash(file_upload.name + pdf_id))}"
        logger.info(f"Creating vector DB with collection name: {collection_name}")

        embeddings = OllamaEmbeddings(model=EmbeddingsStore.emb_model_name)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=collection_name
        )
        logger.info("Vector DB created with persistent storage")
        return vector_db


    @staticmethod
    def delete_vector_db(vector_db: Optional[Chroma]) -> None:
        """
        Delete the vector database and clear related session state.

        Args:
            vector_db (Optional[Chroma]): The vector database to be deleted.
        """
        logger.info("Deleting vector DB")
        if vector_db is not None:
            try:
                vector_db.delete_collection()
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

    @staticmethod
    def remove_page_number_text(text: str) -> str:
        '''
        Docstring for remove_page_number_text
        
        :param text: Description
        :type text: str
        :return: Description
        :rtype: str
        '''
        text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
        text = re.sub(r'^\s*[-–—]?\s*\d+\s*[-–—]?\s*$', '', text)
        return text.strip()
