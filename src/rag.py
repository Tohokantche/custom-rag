import logging, copy
import streamlit as st
from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage


logger = logging.getLogger(__name__)

class RagLogic:
    
    """
        The actulal RAG logic
        
    """

    def __init__(self, selected_model: str, router_model:str="lfm2.5-thinking:1.2b"):
    
        # Template for the queries router/guardrails
        self.router_template = """
            Accurately classify the query into exactly one category: 
            'retrieval', 'casual', toxic, or 'unsure'. You can use previous interaction as a guidance.

            - 'retrieval': explicit request for external facts/data that requires specific document/database access
            - 'casual':  small talk/greetings/casual chat/personal conversation
            - 'toxic':  hate speech/explicit insult/disrespectful statement/ malicious intention
            - 'unsure':  small talk that are ambiguous or unclear or vague or unrelated to the context
                                                                                                               
            Query: {question}
            Category:"""

        self.chat_model = ChatOllama(model=selected_model, temperature=0.6, top_p=0.95, top_k=20)
        try :
            self.router_model = ChatOllama(model=router_model, temperature=0.1, top_k=50)
        except Exception as e:
            logger.error(f"Unable to load the router model {router_model}: {e}")
            logger.warning(f"Falling back to the chat model {selected_model}.")
            self.router_model = ChatOllama(model=selected_model, temperature=0.1)
            st.warning('Using chat model as a router. Performance might degrade and response might be slow !', icon="⚠️")

        self.chat_context = None
        self.sources = None
        self.response= None

    def retrieve_doc(
        self,
        question: str,
        pdfs_dict: Dict[str, Dict],
        config : Dict = None
    ):
        
        """Query across multiple PDFs with source attribution."""
        logger.info(f"Processing question across {len(pdfs_dict)} PDFs: {question}")
         # Query prompt
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are a helpful AI language model assistant. Your task is to generate exactly 2
            different versions of the given user question to retrieve only relevant documents from a vector 
            database. By generating multiple semantically faithful perspectives on the user question, your 
            goal is to help the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines, and stick as much as you can to the 
            semantic information of the original question without hallucinating.
            Original question: {question}"""
        )

        # Retrieve from ALL PDF collections
        all_retrieved_docs = []
        temp_querry_docs_pairs =[]
        temp_sources = []
        for pdf_id, pdf_data in pdfs_dict.items():
            vector_db = pdf_data["vector_db"]

            # Setup Multi querry retriever
            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(search_kwargs={"k": 5}),
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
                    bm25_retriever.k = 5 

                    # Weighted fusion of the retrivers 
                    ensemble_retriever = EnsembleRetriever(
                                retrievers=[bm25_retriever, retriever],
                                weights=[0.35, 0.65]
                            )
                    docs = ensemble_retriever.invoke(question) 
                else:
                    docs = retriever.invoke(question)

                logger.info(f"Retrieved {len(docs)} documents from {pdf_data['name']}")
                
                # Ensure metadata is collected for referencing
                for doc in docs:
                    if "pdf_name" not in doc.metadata:
                        doc.metadata["pdf_name"] = pdf_data["name"]
                    if "pdf_id" not in doc.metadata:
                        doc.metadata["pdf_id"] = pdf_id
                    if config['re_ranker'] and config :
                        temp_querry_docs_pairs.append((question, doc.page_content))
                        temp_sources.append(doc.metadata)
                all_retrieved_docs.extend(docs)
                
            except Exception as e:
                logger.warning(f"Error retrieving from {pdf_data['name']}: {e}")

        logger.info(f"Total documents retrieved: {len(all_retrieved_docs)}")

        # Format context with source labels
        context_parts = []

        # Re-rankers the retrieved documents
        if config['re_ranker'] and config:

            logger.info(f"Re-ranking the retreived documents")
            re_rank_documents = config['re_ranker'].get_ranked_document(temp_querry_docs_pairs, temp_sources)
            for score, doc, source in re_rank_documents:

                # Keep only documents/chucks with similarity greater or equal to 0.35
                if score >= 0.35:
                    context_parts.append(f"[Source: {source.get('pdf_name', 'Unknown')}]\n{doc}\n")
        else:
            # Keep Top 10 chunks
            for doc in all_retrieved_docs if len(all_retrieved_docs) < 10 else all_retrieved_docs[:10]: 
                source = doc.metadata.get("pdf_name", "Unknown")
                context_parts.append(f"[Source: {source}]\n{doc.page_content}\n")
        return  "\n---\n".join(context_parts), all_retrieved_docs
        
    
    def process_question_multi_pdf(
        self,
        question: str,
        pdfs_dict: Dict[str, Dict],
        chat_context: List,
        config : Dict = None
    ):
        self.chat_context = chat_context
        formatted_context, all_retrieved_docs = None, None

        logger.info(f"Classifying the user query: {question}")
        router_messages = [
                    SystemMessage(content=" You are a user intent detector. Your role is to accurately " \
                    "classify the user intent without hallucinating.")
                ]
        
        # By default classify queries as casual
        router_response = "casual"

        # Only route queries with more than 12 characters to prevent unecessary computation
        if not (len(question) < 12):

            # copy chat history to serve as a context for the query classification 
            temp_chat = copy.deepcopy(chat_context)
            router_messages.extend(temp_chat)
            router_messages.append(HumanMessagePromptTemplate.from_template(self.router_template))
            router_chain = (
                {"question": lambda x: x["question"]}
                | ChatPromptTemplate.from_messages(router_messages)
                | self.router_model 
                | StrOutputParser()
            )
            router_response = router_chain.invoke({"question": question})
        if "retrieval" in router_response:
            logger.info(f"{question} is clasified as a retrival query.")
            formatted_context, all_retrieved_docs = self.retrieve_doc(question, pdfs_dict, config)
            
            # RAG prompt with source awareness
            template = """Answer the question based ONLY on the following context from one or multiple PDF documents.
                Please only provide HIGHLY relevant information and do not hallucinate.
                Each section is marked with its source document.

                When answering:
                1. Cite only the relevant source document name for each piece of information
                2. If information comes from multiple sources, mention all relevant sources
                3. If sources contradict, note the discrepancy and cite both

                Context:
                {context}

                Question: {question}

                Answer (include source citations):"""

        elif "casual" in router_response:
            logger.info(f"{question} is clasified as a casual query.")
            template = """
                Respond casually or based on the previous interactions to: {question}
            """
        elif "toxic" in router_response:
            logger.info(f"{question} is clasified as a toxic/malicious query.")
            template = """
                You are a helpful assistant. Respond vaguely based on previous interactions to : {question}
            """
        else:
            logger.info(f"{question} is clasified as unsure query intent.")
            template = """
                 Respond or ask politely for clarification about the query : {question}
            """
            
        self.chat_context.append(HumanMessagePromptTemplate.from_template(template))
        decision_chain = ChatPromptTemplate.from_messages(self.chat_context) | self.chat_model

        response_chain = (
            {"context": lambda x: formatted_context, "question": lambda x: x["question"]} 
            | decision_chain
            | StrOutputParser()
            )

        logger.info("Generating response with source attribution")
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
        
    