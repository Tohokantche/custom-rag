"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

from src.ui.chat import load_chat_ui
import logging, sys, os

# Suppress torch warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    # {%(pathname)s:%(lineno)d}
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log"), mode='w'), 
        logging.StreamHandler(sys.stdout)         
    ], 
)

def main() -> None:
    """
    Main function to run the Streamlit application. This method is the entry-point to the UI and its logic
    """
    
    load_chat_ui()

if __name__ == "__main__":
    main()
