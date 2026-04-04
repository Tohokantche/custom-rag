from pydantic import BaseModel

class PromptManager(BaseModel):
    """ 
        Prompt template used across the application
    """

    ROUTER_SYSTEM_PROMPT : str = " You are a user intent detector. Your role is to accurately " \
                    "classify the user intent without hallucinating."
    
    ROUTER_PROMPT_TEMPLATE : str =  """
            Accurately classify the query into exactly one category: 
            'retrieval', 'casual', toxic, or 'unsure'. You must use previous interactions as a guidance to classify the query.

            - 'retrieval': explicit request for external concepts/facts/data that requires specific document access
            - 'casual':  casual or small talk/greetings, or personal conversation
            - 'toxic':  hate speech/explicit, or insult/disrespectful statement/malicious intention
            - 'unsure':  small talk that are ambiguous, or unclear, or vague
                                                                                                               
            Query: {question}
            Category: """
    
    ROUTER_RETRIEVAL_PROMPT_TEMPLATE :str = """Answer the question based ONLY on the following context from one or multiple documents.
                Please only provide HIGHLY relevant information and do not hallucinate.
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
                Respond to {question}.
            """
    
    ROUTER_TOXIC_PROMPT_TEMPLATE : str = """
                You are a helpful assistant. Respond vaguely to : {question}"""
    
    ROUTER_UNSURE_PROMPT_TEMPLATE : str = """
                 You can access some external documents. Respond politely, or request politely for clarification about the query 
                 using, or not using previous interactions : {question}"""

    MULTI_QUERY_PROMPT_TEMPLATE: str = """
            You are a helpful AI language model assistant. Your task is to generate exactly 3
            different versions of the given user question to retrieve only relevant documents from a vector 
            database. By generating multiple semantically faithful perspectives on the user question, your 
            goal is to help the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines, and stick as much as you can to the 
            semantic information of the original question without hallucinating. You may use the chat 
            history to contextualize the query, but do not change the semantic.

            'Original question':
            {question}"""

