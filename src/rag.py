import logging
from typing import Dict, List, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RagLogic:

    @staticmethod
    def process_question_multi_pdf(
        question: str,
        pdfs_dict: Dict[str, Dict],
        selected_model: str,
        chat_context,
        config : Dict = None
    ) -> Tuple[str, List[Dict]]:
        
        """Query across multiple PDFs with source attribution."""
        
        logger.info(f"Processing question across {len(pdfs_dict)} PDFs: {question}")
        llm = ChatOllama(model=selected_model)

        # Query prompt
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are a helpful AI language model assistant. Your task is to generate exactly 2
            different versions of the given user question to retrieve only relevant documents from a vector 
            database. By generating multiple semantically faithful perspectives on the user question, your 
            goal is to help the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines, and stick as much as you can to the 
            semantic of the original question without hallucinating.
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
                vector_db.as_retriever(search_kwargs={"k": 7}),
                llm,
                prompt=QUERY_PROMPT
            )
            try:
                if config['h_search'] and config :
                    logger.info(f"Initialising the hybrid search strategy")
                    # Setup BM25 Retriever (Keyword search)
                    bm25_retriever = BM25Retriever.from_documents([Document(page_content=doc) for doc in vector_db.get()['documents']])
                    bm25_retriever.k = 7 

                    # Weighted fusion of the retrivers 
                    ensemble_retriever = EnsembleRetriever(
                                retrievers=[bm25_retriever, retriever],
                                weights=[0.3, 0.7]
                            )
                    docs = ensemble_retriever.invoke(question) 
                else:
                    docs = retriever.invoke(question)

                logger.info(f"Retrieved {len(docs)} documents from {pdf_data['name']}")
                # Ensure metadata
                for doc in docs:
                    if "pdf_name" not in doc.metadata:
                        doc.metadata["pdf_name"] = pdf_data["name"]
                    if "pdf_id" not in doc.metadata:
                        doc.metadata["pdf_id"] = pdf_id
                    if config['re_ranker'] and config :
                        temp_querry_docs_pairs.append((question, doc.page_content))
                        temp_sources.append(doc.metadata.get("pdf_name", "Unknown"))
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
                    context_parts.append(f"[Source: {source}]\n{doc}\n")
        else:
            for doc in all_retrieved_docs[:10]:  # Top 10 chunks
                source = doc.metadata.get("pdf_name", "Unknown")
                context_parts.append(f"[Source: {source}]\n{doc.page_content}\n")

        formatted_context = "\n---\n".join(context_parts)

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

        #prompt = ChatPromptTemplate.from_template(template)
        #chat_context.append(SystemMessage(content="You are a helpful assistant."))
        chat_context.append(HumanMessagePromptTemplate.from_template(template))
        chat_messages= ChatPromptTemplate.from_messages(chat_context)

        chain = (
            {"context": lambda x: formatted_context, "question": lambda x: x}
            | chat_messages #prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        logger.info("Generated response with source attribution")

        # Extract source details
        source_details = [
            {
                "pdf_name": doc.metadata.get("pdf_name"),
                "pdf_id": doc.metadata.get("pdf_id"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            for doc in all_retrieved_docs[:12]
        ]
        return response, source_details
    
