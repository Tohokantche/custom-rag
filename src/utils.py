import logging
from typing import Tuple, Any, Union, List
from threading import Thread
import streamlit as st
import numbers
import subprocess
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


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
    
    def use_generation_model(self, prompt="Why is the sky blue?"):
        logger.info(f"Using generation model: {self.model_name}")
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt, stream=False)
            return response['response']
        except Exception as e:
            # A generation-only model will likely fail if forced into an embedding call,
            # but a general LLM can produce embeddings through the 'embed' function
            # (though typically less specialized than dedicated embedding models)
            logger.info(f"Error generating text with {self.model_name}: {e}")
            return None

    # --- Embedding Model ---
    def use_embedding_model(self, input_text="The quick brown fox jumps over the lazy dog."):
        logger.info(f"Using embedding model: {self.model_name}")
        try:
            # The 'embed' function is designed for generating vector embeddings
            embeddings = self.client.embed(model=self.model_name, input=input_text)
            # The result is a list of floating point numbers
            return embeddings['embeddings'][0]
        except Exception as e:
            logger.info(f"Error generating embeddings with {self.model_name}: {e}")
            return None
    
    def run(self):
        emb_models = False
        gen_data = self.use_generation_model()
        embedding = self.use_embedding_model()
        if embedding and self.is_vector_numeric(embedding):
            emb_models = True
        elif gen_data and not self.is_vector_numeric(gen_data):
            emb_models = False
        elif self.is_vector_numeric(gen_data):
            emb_models = True
        self.return_value = emb_models

class DownloadModels(Thread):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.command = ["ollama", "pull", model_name]
        self.return_value = False
    
    def execute_command(self, timeout: int=180) -> Union[subprocess.CompletedProcess[str], None]:
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
            logger.error(f"Error: Command '{command[0]}' not found - {e}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Process failed with exit code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        return result
    
    def run(self):
        result = self.execute_command(command)
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
            
            self.model = CrossEncoder(model_name, 
                trust_remote_code=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

        def get_ranked_document(self, query_documents: List[Tuple[str, str]] = None, sources : List[str] = None):
            if not query_documents and not sources:
                scores = self.model.predict(query_documents)
                logger.info(f"Re-rank documents score : {scores}")
                scored_documents = sorted(zip(scores, query_documents, sources), key=lambda x : x[0], reverse= True )
                return scored_documents 
            return None

def render_config():
    """Render and handle chunck configuration.
    
    The sidebar allows users to:
    1. Choose chunk size value
    2. Choose chunk overlap value
   
    Returns:
        dict: Contains chunk configuration
    """
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        with st.expander("🧱 Chunk Setting", expanded=False):
            chunk_size = st.number_input("Enter chunk size:", value=1500, min_value=256, max_value=10000, step=1, 
                                          format="%i", help="Specify chunk size")
            chunk_overlap = st.number_input("Enter chunk overlap:", value=100, min_value=10, max_value=500, 
                                             step=1, format="%i", help="Specify chunk overlap")
            st.divider
            re_ranker = st.checkbox("Use Re-ranker")
            if re_ranker:
                st.warning('This might really slow down the RAG response time, as this demo run on free tier CPU cloud \
                           with very limited ressources!', icon="⚠️")
    return {"chunk_size": str(chunk_size), "chunk_overlap": str(chunk_overlap), "re_ranker": re_ranker}


def generate_pdf_id(file_upload) -> str:
    """Generate unique ID for PDF."""
    #timestamp = datetime.now().isoformat()
    return f"pdf_{abs(hash(file_upload.name))}" 


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    gen_models = []
    emb_models = []
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
            # SEperate Gen Models from Embeddings ones
            # model_threads = [LoadModelsThread(ollama.Client(), model_name) for model_name in model_names]
            # for thread in model_threads:
            #     thread.start()
            # for thread in model_threads:
            #     thread.join()
            # for i, thread in enumerate(model_threads):
            #     if thread.return_value:
            #         emb_models.append(thread.model_name)
            #     else:
            #         gen_models.append(thread.model_name)

            # logger.info(f"Embedding models --- {emb_models}")
            # logger.info(f"Generation models --- {gen_models}")
        else:
            # Fallback for any other format
            model_names = tuple()
            ## Download one chat, on embedding and one re-ranker models 
            
        logger.info(f"Extracted model names: {model_names}")
        # return model_names
        return tuple(model_names)
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()
    

def execute_command(command: List[str], timeout: int=15) -> Union[subprocess.CompletedProcess[str], None]:
    """ Method for executing a shell command """
    result = None
    try:
        logger.info(f"Running : {command}")
        result = subprocess.run(
        command,
        #timeout=timeout,
        check=True,
        capture_output=True,
        text=True 
        )
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {e.timeout} seconds.")
        logger.error(f"Captured stdout: {e.stdout}")
        logger.error(f"Captured stderr: {e.stderr}")
    except FileNotFoundError as e:
        logger.error(f"Error: Command '{command[0]}' not found - {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Process failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return result


if __name__ == "__main__":
    # nomic-embed-text:latest, hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q6_K , 
    command = ["ollama", "pull", "smollm:135m"]
    logger.info(command)
    result = execute_command(command)
    logger.info(result.stderr)
    if ("success" in result.stderr) and result:
        logger.info("Sucessfully download the models")
    else:
        logger.info("Unable to download the file")