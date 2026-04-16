import sqlite3
import logging
import json, os
from concurrent.futures import ThreadPoolExecutor
from langchain_ollama import ChatOllama
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import torch
from functools import lru_cache
from src.prompt import PromptManager
from src.utils import detect_device


PERSIST_DIRECTORY = os.path.join("data", "vectors")
logger = logging.getLogger(__name__)


class QueryRouter:
    """
    A simple query router that decides if a given query requires specific internal document 
    or a general knowledge to be answered.
    """

    prompt_manager : PromptManager = PromptManager()

    def __init__(self, db_path='documents.db', max_workers=4, model_name='qwen3:1.7b'):
        self.db_path = os.path.join(PERSIST_DIRECTORY, db_path) 
        # llama3.2:1b
        self.model_name = model_name
        self.max_workers = max_workers
        self.threshold = 0  # Default: BM25 scores are negative; -1.0 is a solid match
        # Initialize with structured output for fast parsing
        self.query_gen_model = ChatOllama(
            model=model_name,
            temperature=0.6, 
            top_p=0.95, 
            top_k=20,
            keep_alive=-1
            ) 
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sentence_model.to(detect_device()["device"])
        self.sentence_model.compile(dynamic=True)
        self._init_tables()
        logger.info("Successfully initialised the Query Router class and its models!")

    
    def tune_threshold(self):
        """Analyzes historical logs to find the optimal BM25 threshold."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get scores for queries that were routed to internal_docs
            cursor.execute("SELECT score FROM query_logs WHERE route = 'internal_docs' AND score < 0")
            scores = [row[0] for row in cursor.fetchall()]
        
        if not scores:
            logger.warning("Not enough data to tune. Keeping default threshold.")
            return self.threshold

        # Calculate the 75th percentile (or mean) of successful scores
        # We want a threshold that captures most successful hits but filters noise
        new_threshold = np.percentile(scores, 75) 
        self.threshold = new_threshold
        logger.info(f"Threshold tuned to: {self.threshold:.4f}")
        return self.threshold


    def _init_tables(self):
        """Initializes the BM25-enabled virtual table and log table."""
        with sqlite3.connect(self.db_path) as conn:
            # FTS5 virtual table for BM25 search. 'porter' handles stems like 'running' -> 'run'
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS document_index 
                USING fts5(filename, keywords, tokenize='porter')
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    keywords TEXT,
                    route TEXT,
                    filename TEXT,
                    score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()


    def ingest_document (self, file_path: str):
        """Extracts keywords and upserts into the FTS5 virtual table."""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info("Ingesting document {file_path}")
        if len(pages) <=10:
            text = " ".join([p.page_content for p in pages if p])
        else:
            text = " ".join([p.page_content for p in (pages[:7] + pages[-3:]) if p])
        logger.info(f"Doc length : {len(text)}")
        kw_model = KeyBERT(model=self.sentence_model)
        keywords = set()
        left_id = 0
        logger.info("Extracting keywords from the document ...")
        for right_id in range(512 ,len(text), 432):          
            curr_keywords = kw_model.extract_keywords(text[left_id:right_id].strip().lower())
            logger.info(curr_keywords)
            [keywords.add(k) for k,score in curr_keywords if score >= 0.35]
            left_id = right_id - 80

        logger.info(f"I extracted {len(keywords)} keywords which are : {keywords}")
        keywords = json.dumps(list(keywords)).replace("[", "").replace("]", "").replace('"', "").replace(",", "").strip()
        logger.info(f"Parse keywords -- {keywords}")
        with sqlite3.connect(self.db_path) as conn:
            # FTS5 tables don't support 'UNIQUE' constraints; we manually delete then insert
            conn.execute("DELETE FROM document_index WHERE filename = ?", (file_path,))
            conn.execute("INSERT INTO document_index (filename, keywords) VALUES (?, ?)", 
                         (file_path, keywords))
            conn.commit()
        return file_path

    @lru_cache(maxsize=1000)
    def _extract_keywords_cached(self, query):
        """Skip the LLM step for identical queries via in-memory caching."""
        return self._extract_keywords(query)

    def _extract_keywords(self, query):
        """ Extract keywords form the user query"""
        DEFAULT_QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=QueryRouter.prompt_manager.ROUTER_PROMPT_TEMPLATE,
        )
        query_chain = (
            {'question':  RunnablePassthrough()}
            | DEFAULT_QUERY_PROMPT
            | self.query_gen_model
            | StrOutputParser()
        )
        response = query_chain.invoke(query)
        logger.info(f"Generated queries:\n {response}")
        kw_model = KeyBERT(model=self.sentence_model)
        keywords = kw_model.extract_keywords(response, stop_words=None)
        return list({k for k,score in keywords})
    
    def route(self, query: str):
        """Routes using BM25 ranking for sub-millisecond keyword lookup."""
      
        q_terms = self._extract_keywords_cached(query)
        search_query = " OR ".join(q_terms)
        matches = []
        best_score = 0.0
    
        if q_terms:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT filename, bm25(document_index) as score 
                    FROM document_index 
                    WHERE document_index MATCH ? 
                    ORDER BY score ASC LIMIT 5
                ''', (search_query,))
                results = cursor.fetchall()
                if results:
                    matches = [r[0] for r in results]
                    best_score = results[0][1]

        # Use a relevance threshold: only route to docs if the score is strong (e.g. < -0.5)
        logger.info(f"matches:{matches} --- best_score:{best_score}")
        route = "internal_docs" if (matches and best_score < self.threshold) else "general_knowledge"
        
        # Log analytics with the BM25 score
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO query_logs (query, keywords, route, filename, score) VALUES (?, ?, ?, ?, ?)",
                        (query, json.dumps(q_terms), route, json.dumps(matches), best_score))
            conn.commit()
        return {"route": route, "matches": matches, "relevance_score": best_score}
    
    
    def batch_ingest(self, folder_path):
        """Parallel ingestion using the pool."""
        if not folder_path:
            return
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(executor.map(self.ingest_document, files))

    
    def remove_document(self, file_path: str):
        """Removes a document and its keywords from the BM25 index."""
        if not file_path:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT filename FROM document_index WHERE filename = ?", (file_path,))
                exists = cursor.fetchone()
                if not exists:
                    logger.info(f"Document not found in index: {file_path}")
                    return False
                # Perform the deletion from the FTS5 virtual table
                conn.execute("DELETE FROM document_index WHERE filename = ?", (file_path,))
                conn.commit()
            logger.info(f"Successfully removed from index: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False


    def clear_all_indexes(self):
        """Wipes the entire search index (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM document_index")
            conn.commit()
            logger.info("Entire document index has been cleared.")
    
    def clear_all_queries(self):
        """Wipes the entire auery logs (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM query_logs")
            conn.commit()
            logger.info("Entire query logs have been cleared.")


# if __name__== "__main__":
  
#     q_router = QueryRouter()
#     q_router.batch_ingest("/Users/aurele/Desktop/test/")
#     logger.info(q_router.route("What is instance normalisation ?"))
    
