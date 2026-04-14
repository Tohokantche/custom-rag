import logging, copy, os
from typing import Dict, List
from src.prompt import PromptManager
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from src.query_router import QueryRouter
import concurrent.futures
from langchain_core.prompts import (
    ChatPromptTemplate, 
    PromptTemplate, 
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda,
    ) 

logger = logging.getLogger(__name__)

class RagLogic:
    """
        THE ACTUAL RAG LOGIC
        
    """
    prompt_manager : PromptManager = PromptManager()
    

    def __init__(self, selected_model: str):
        
       
        # logger.info(f"Using {router_model} for router model")
        self.chat_model = ChatOllama(
            model=selected_model, 
            temperature=0.6, 
            top_p=0.95, 
            top_k=20
            )
        logger.info(f"Using {selected_model} for chat model")
        self.query_router : QueryRouter = QueryRouter(model_name=selected_model)
        logger.info(f"Using {selected_model} for router model")
        self.prune_chat_context = None
        self.sources = None
        self.response= None

    def set_chat_model(self, chat_model_name : str = None) -> bool:
        if chat_model_name:
            logger.info(f"Set {chat_model_name} as the chat model")
            self.chat_model = ChatOllama(
            model=chat_model_name, 
            temperature=0.6, 
            top_p=0.95, 
            top_k=20
            )
            return True
        return False
    
    def set_router_model(self, router_model_name : str = None) -> bool:
        if router_model_name:
            logger.info(f"Set {router_model_name} as the router model")
            self.query_router = QueryRouter(model_name = router_model_name)
            return True
        return False

    def format_chat_history(self, chat_history):
        """ 
        Turn List of chat history into a String and prune special characters
        return a chat history as a string
        """
        if chat_history:
            formated_chat_history = "\n".join([f"{chat.type.capitalize()}: {chat.content}" 
                            for chat in chat_history if chat.type.capitalize() != "system"])
            # Removing curly brace to prevent potential Langchain prompttemplate processing error 
            return formated_chat_history.replace("{", "").replace("}", "")
        return None


    def retrieve_doc(self, pdf_id, pdf_data, config, QUERY_PROMPT, question):
        """
        Retrieve relevant chunks from a single PDFs documents with source attribution.
        return pairs of retrieved chunks and query, corresponding source, and all retrived chucks
        """
        # Retrieve from current PDF collections
        curr_retrieved_docs = []
        curr_querry_docs_pairs =[]
        curr_sources = []

        vector_db = pdf_data["vector_db"]

        # Setup Multi querry retriever
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 3}),
            self.chat_model,
            prompt=QUERY_PROMPT
        )
        try:
            if config['h_search'] and config :
                logger.info(f"Initialising the hybrid search strategy")
                
                # Setup BM25 Retriever (Keyword search)
                bm25_retriever = BM25Retriever.from_documents(
                    [Document(page_content=doc, metadata=metadata) for doc, metadata 
                        in zip(vector_db.get()['documents'], vector_db.get()['metadatas'])])
                bm25_retriever.k = 3 

                # Weighted fusion of the retrievers a.k.a hybrid search
                ensemble_retriever = EnsembleRetriever(
                            retrievers=[bm25_retriever, retriever],
                            weights=[0.35, 0.65]
                        )
                docs = ensemble_retriever.invoke(question)
            else:
                docs = retriever.invoke(question)

            logger.info(f"Retrieved {len(docs)} documents from {pdf_data['name']}")
            
            # Ensuring metadata is collected for referencing
            for doc in docs:
                if "pdf_name" not in doc.metadata:
                    doc.metadata["pdf_name"] = pdf_data["name"]
                if "pdf_id" not in doc.metadata:
                    doc.metadata["pdf_id"] = pdf_id
                if config['re_ranker'] and config:
                    # Temporaly separately  store querry_doc_pairs and sources to match the re-ranker input format
                    curr_querry_docs_pairs.append((question, doc.page_content))
                    curr_sources.append(doc.metadata)
            curr_retrieved_docs.extend(docs)
            
        except Exception as e:
            logger.warning(f"Error retrieving from {pdf_data['name']}: {e}")

        return curr_querry_docs_pairs, curr_sources, curr_retrieved_docs
    

    def retrieve_all_doc(
        self,
        question: str,
        pdfs_dict: Dict[str, Dict],
        router_file_match: List,
        config : Dict = None
    ): 
        """
        Retrieve relevant chunks across multiple PDFs documents with source attribution.
        return retrieved PDFs chunk and a formatted bundel of chunk and sources
        """
        
        logger.info(f"Processing question across {len(pdfs_dict)} PDFs: {question}")
        logger.info(f"Router match consists of : {len(router_file_match)} PDFs")
        # Query prompt for query expansion. Chat history is appended for query contextualization
        template = RagLogic.prompt_manager.MULTI_QUERY_PROMPT_TEMPLATE
        
        QUERY_PROMPT = PromptTemplate.from_template(template)
        
        # Retrieve from ALL PDF collections
        all_retrieved_docs = []
        all_querry_docs_pairs = []
        all_sources = []

        # Multithreading of the retrieval task to reduce latency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                self.retrieve_doc, 
                pdf_id, 
                pdf_data, 
                config, 
                QUERY_PROMPT, 
                question) : pdf_id for pdf_id, pdf_data in pdfs_dict.items() if pdf_data['name'] in router_file_match}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    all_querry_docs_pairs.extend(results[0])
                    all_sources.extend(results[1])
                    all_retrieved_docs.extend(results[2])
                except Exception as error_msg:
                    logger.error(f"Unexpected error ocuurs with {futures[future]} which is cause by :{error_msg}")

        logger.info(f"Total documents retrieved: {len(all_retrieved_docs)}")

        # Initialize context (doc + source) formating variable
        context_parts = []

        # Powerfull Elo Re-rankers to re-rank the retrieved documents (work best on GPU, very slow on CPU)
        if config['re_ranker'] and config:
            logger.info(f"Re-ranking the {len(all_querry_docs_pairs)} retreived documents !")
            re_rank_documents = config['re_ranker'].get_ranked_document(all_querry_docs_pairs, all_sources)
            for score, doc, source in re_rank_documents:
                # Keep only documents/chucks with similarity greater or equal to 0.65
                if score >= 0.65:
                    context_parts.append(f"[Source: {source.get('pdf_name', 'Unknown')}]\n{doc}\n")
        else:
            # Keep Top 10 chunks
            for doc in all_retrieved_docs if len(all_retrieved_docs) < 10 else all_retrieved_docs[:10]: 
                source = doc.metadata.get("pdf_name", "Unknown")
                context_parts.append(f"[Source: {source}\n{doc.page_content}\n")
        return  "\n---\n".join(context_parts), all_retrieved_docs
    
    
    def process_question_multi_pdf(
        self,
        question: str,
        pdfs_dict: Dict[str, Dict],
        chat_context: List,
        config : Dict = None
    ):
        """
        Process the user query by acessing all available documents
        return assistant/AI response asynchonously
        """
        self.prune_chat_context = copy.deepcopy(
            chat_context if len(chat_context) < 7 else chat_context[-6:]
            )
        formatted_context, all_retrieved_docs = None, None
        # Routing the user query
        router_response = self.query_router.route(question)

        if 'internal_docs' in router_response['route']:
            match_files_path = [os.path.basename(path) for path in router_response['matches']]
            logger.info(f"Router has matched '{question}' to these PDFs: {match_files_path}")
            logger.info(f"'{question}' is clasified as a retrival query.")
            formatted_context, all_retrieved_docs = self.retrieve_all_doc(question, pdfs_dict, match_files_path, config)
            # RAG prompt with source awareness
            router_category_template = RagLogic.prompt_manager.ROUTER_RETRIEVAL_PROMPT_TEMPLATE
            logger.info("Generating response with source attribution")
        else:
            router_category_template = RagLogic.prompt_manager.ROUTER_CASUAL_PROMPT_TEMPLATE
            logger.info("Generating response")

        self.prune_chat_context.append(HumanMessagePromptTemplate.from_template(router_category_template))
        chat_context.append(HumanMessage(content=question))
        decision_chain = ChatPromptTemplate.from_messages(self.prune_chat_context) | self.chat_model

        response_chain = (
            {"context": lambda x: formatted_context, "question": lambda x: x["question"]} 
            | decision_chain
            | StrOutputParser()
            )
        
        # Returning the assistance response
        response = ""
        for chunk in response_chain.stream({"question": question}):
                response+= chunk
                yield chunk

        source_details = [
                    {
                        "pdf_name": doc.metadata.get("pdf_name"),
                        "pdf_id": doc.metadata.get("pdf_id"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "chunk_value": doc.page_content
                    }
                    for doc in all_retrieved_docs
                ] if all_retrieved_docs else None
        
        self.response = response
        self.sources = source_details
        