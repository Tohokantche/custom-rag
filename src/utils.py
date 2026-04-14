import logging
from typing import Tuple, Any, Union, List, Dict
from threading import Thread
import streamlit as st
import numbers
import subprocess
import torch
import ollama
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

class DownloadModels(Thread):
    """
    
    """
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.command = ["ollama", "pull", model_name]
        self.return_value = False
    
    def execute_command(self, timeout: int=600) -> Union[subprocess.CompletedProcess[str], None]:
        """ Method to download the embedding and the chat models"""
        result = None
        try:
            logger.info(f"Running : {self.command}")
            result = subprocess.run(
            self.command,
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True 
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {e.timeout} seconds.")
            logger.error(f"Captured stdout: {e.stdout}")
            logger.error(f"Captured stderr: {e.stderr}")
        except FileNotFoundError as e:
            logger.error(f"Error: Command '{self.command[0]}' not found - {e}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Process failed with exit code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        return result
    
    def run(self):
        result = self.execute_command(self.command)
        if ("success" in result.stderr) and result:
            self.return_value=True


class LoadReRanker(Thread):

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.return_value = None
    def run(self):
            self.return_value=self.ReRanker(self.model_name)

    class ReRanker():
        def __init__(self, model_name):
            
            device = detect_device()
            self.model = CrossEncoder(
                model_name,
                device = device["name"],
                trust_remote_code = True
                )
            # conert to brain floating point to speed up inference
            self.model.model.to(dtype=torch.bfloat16)
            # Use torch.compile to speed up inference
            if device["name"] in ["mps", "cuda"]:
                self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
            else:
                self.model.model = torch.compile(self.model.model, backend="inductor")

        def get_ranked_document(self, query_documents: List[Tuple[str, str]] = None, sources : List[Dict[str,str]] = None):
            logger.info("Re-ranking retrieved documents")
            if query_documents and sources:
                scores = self.model.predict(query_documents)
                logger.info(f"Re-rank retrieved documents score : {scores}")
                scored_documents = sorted(zip(scores, query_documents, sources), key=lambda x : x[0], reverse= True )
                return scored_documents 
            return None

class LoadModelsThread(Thread):
    def __init__(self, client, model_name):
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.return_value = None
    
    def is_vector_numeric(self, vector):
        for item in vector:
            if not isinstance(item, numbers.Number):
                return False
        return True
    
    def use_generation_model(self, prompt="hello there!"):
        logger.info(f"Testing model for gen: {self.model_name}")
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt, stream=False)
            logger.info(f"Testing modelfor gen: {self.model_name} -- response : {response['response']}")
            return response['response']
        except Exception as e:
            # A generation-only model will likely fail if forced into an embedding call,
            # but a general LLM can produce embeddings through the 'embed' function
            # (though typically less specialized than dedicated embedding models)
            logger.error(f"Error generating text with {self.model_name}: {e}")
            return None

    # --- Embedding Model ---
    def use_embedding_model(self, input_text="Why is the sky blue ?"):
        logger.info(f"Testing model for embedd: {self.model_name}")
        try:
            # The 'embed' function is designed for generating vector embeddings
            embeddings = self.client.embed(model=self.model_name, input=input_text)
            # The result is a list of floating point numbers
            logger.info(f"Testing for embedd: {self.model_name} -- response : {embeddings['embeddings'][0]}")
            return embeddings['embeddings'][0]
        except Exception as e:
            logger.info(f"Error generating embeddings with {self.model_name}: {e}")
            return None
    
    def run(self):
        # Spearate generation models from embeddings ones by testing their output
        emb_models = False
        gen_data = self.use_generation_model()
        embedding = self.use_embedding_model()
        if embedding and self.is_vector_numeric(embedding):
            emb_models = True
        if gen_data and not self.is_vector_numeric(gen_data):
            emb_models = False
        self.return_value = emb_models


def detect_device():
    """ Detect host device"""

    name = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = "cuda"
        logger.info(f"Detected NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "mps"
        logger.info("Detected Apple Silicon (Mac) GPU via MPS")
    else:
        device = torch.device("cpu")
        name = "cpu"
        logger.info("No GPU detected, using CPU")
    return {"device": device, "name": name}

def extract_model_names(models_info: Any) -> Dict[str,Tuple[str, ...]]:
    """
    Extract model names from the provided models information 
    and identifying model type.
    return a dictionnary of embedded and generative model names.
    """

    logger.info("Extracting model names from models_info")
    # Initialise list for generation and embeddings models
    gen_models = []
    emb_models = []
    try:
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
            
            # Seperate Gen Models from Embeddings ones
            model_threads = [LoadModelsThread(ollama.Client(), model_name) for model_name in model_names]
            logger.info("Identifying extracted models type: embedding or generative")
            for thread in model_threads:
                thread.start()
            for thread in model_threads:
                thread.join()
            for thread in model_threads:
                if thread.return_value:
                    emb_models.append(thread.model_name)
                else:
                    gen_models.append(thread.model_name)

            logger.info(f"Embedding models names --- {emb_models}")
            logger.info(f"Generation models names --- {gen_models}")
            
        logger.info(f"All extracted model names: {model_names}")
        # return model_names
        return {'emb_models': tuple(emb_models) , 'gen_models': tuple(gen_models)}
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return {'emb_models': tuple(emb_models) , 'gen_models': tuple(gen_models)}
    
def render_config(emb_models_name : List[str], router_models_name: List[str]):
    """
    Render and handle chunck configuration.
        The sidebar allows users to:
        1. Choose chunk size value
        2. Choose chunk overlap value
    Returns:
        dict: Contains chunk configuration
    """
    
    st.markdown("### ⚙️ Configuration")
    with st.expander("🧱 Retrieval Settings", expanded=True):
        
        selected_router_model = st.selectbox(
            "Pick a query router model:",
            router_models_name,
            key="router_model_select"
        )

        selected_emb_model = st.selectbox(
            "Pick an embedding model:",
            emb_models_name,
            key="emb_model_select"
        )
        chunk_size = st.number_input(
            "Enter chunk size:", 
            value=1500, 
            min_value=256, 
            max_value=10000, 
            step=2, 
            format="%i", 
            help="Specify data chunk size"
            )
        chunk_overlap = st.number_input(
            "Enter chunk overlap:", 
            value=80, 
            min_value=10, 
            max_value=200, 
            step=2, 
            format="%i", 
            help="Specify data chunk overlap"
            )
        st.divider()
        h_search = st.checkbox(
            "Use hybrid search", 
            value = True,
            help="Combine semantic (default) with keyword search"
            )
        re_ranker = st.checkbox(
            "Use re-ranker",
            help="Re-rank the retrieved documents for highly accurate RAG. We are using CrossEncoder re-ranker !"
            )
        if re_ranker:
            if not torch.cuda.is_available():
                st.warning('Using CrossEncoder-based re-ranker! Model response time might be very slow, as this demo run on CPU !', icon="⚠️")
            else:
                st.warning('Using CrossEncoder-based re-ranker! Model response time might be slow !', icon="⚠️")
                
    return {"selected_emb_model": selected_emb_model,
            "selected_router_model":selected_router_model,
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap),
            "re_ranker": re_ranker,
            "h_search": h_search}


def generate_pdf_id(file_upload) -> str:
    """
    Generate unique ID for  a given PDF file.
    """
    return f"pdf_{abs(hash(file_upload.name))}" 


