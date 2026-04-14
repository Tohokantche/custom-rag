from pydantic import BaseModel

class PromptManager(BaseModel):
    """ 
        Prompt template used across the application
    """
      
    ROUTER_PROMPT_TEMPLATE: str = """You are an AI language model assistant. Your task is to generate exactly 
    3 different versions of the given user question to retrieve relevant documents from an index database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some 
    of the limitations of keywords-based search. Provide these alternative questions separated by newlines. 
    Original question: {question}"""

    ROUTER_RETRIEVAL_PROMPT_TEMPLATE :str = """Answer the question based ONLY on the following context from one or multiple 
        documents. Please only provide HIGHLY relevant information and do not hallucinate, or make up content.
        Each section is marked with its source document. And, each document start with its source name: 'Source:'.  

        Use chain-of-thought reasoning:
        1. First, identify which parts of the context are relevant to the question
        2. Analyze the information from each source document
        3. Synthesize the information to form a comprehensive answer
        4. Ensure you cite the source document name for each piece of information
        5. If information comes from multiple sources, mention all relevant sources
        6. If sources contradict, note the discrepancy and cite both sources
        
        Context:
        {context}

        Question:
        {question}

        Answer (include source citations):"""

    ROUTER_CASUAL_PROMPT_TEMPLATE : str = """
        You are a helpful assistant. Respond to : {question}."""
    
    ROUTER_TOXIC_PROMPT_TEMPLATE : str = """
        You are a helpful assistant. Respond vaguely to : {question}"""

    MULTI_QUERY_PROMPT_TEMPLATE: str = """
        You are a helpful AI language model assistant. Your task is to generate exactly 3 different versions 
        of the given user question to retrieve only relevant documents from a vector database. By generating 
        multiple semantically faithful perspectives on the user question, your goal is to help the user overcome 
        some of the limitations of the distance-based similarity search. 
        Provide only 3 alternative questions separated by newlines, and use the chat history to only contextualise 
        the questions. Stick as much as you can to the semantic information of the original question 
        without hallucinating. 

        Original question:
        {question}"""
    
 

