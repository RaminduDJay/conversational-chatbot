import logging
from typing import Dict, List, Optional, Tuple, Union

from agents.rag_functions import get_documents, graded_documents, llm_generation, halucinations_score, answer_grade, requery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
MAX_RETRIEVAL_RETRIES = 3
MAX_ANSWER_RETRIES = 3
MAX_HALLUCINATION_RETRIES = 3


def rag_response(
    query: str, 
    intent: str, 
    output_sentiment: str, 
    documents: List[str], 
    history: Optional[str] = None,
    hallucination_retries: int = MAX_HALLUCINATION_RETRIES
) -> str:
    """Generate an answer using the LLM model with hallucination detection.

    Args:
        query: The user's query text
        intent: The user's intent
        output_sentiment: The desired output sentiment/emotion
        documents: The list of relevant documents for context
        history: Optional conversation history
        hallucination_retries: Number of remaining retries for hallucination detection

    Returns:
        The generated response from the LLM

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If maximum retries are exceeded
    """
    # Input validation
    if not query or not intent or not output_sentiment:
        logger.error("Missing required parameters")
        raise ValueError("Query, intent, and output_sentiment are required")
    
    if not documents:
        logger.warning("Empty documents list provided to rag_response")
        
    # Track retries
    if hallucination_retries <= 0:
        logger.warning("Maximum hallucination retries exceeded")
        return "I'm sorry, but I couldn't generate a reliable answer based on the available information."
    
    try:
        logger.info(f"Generating response for query: {query[:50]}...")
        response, used_docs = llm_generation(
            query=query, 
            intent=intent, 
            output_sentiment=output_sentiment, 
            documents=documents, 
            history=history
        )
        
        # Check for hallucinations
        logger.info("Checking for hallucinations in the generated response")
        score = halucinations_score(documents=used_docs, answer=response)
        
        if str(score).lower() == "no":
            logger.warning(f"Hallucination detected. Retries remaining: {hallucination_retries-1}")
            
            # Generate a new query and retry with decremented counter
            new_query = requery(query=query)
            logger.info(f"Requery generated: {new_query[:50]}...")
            
            return rag_response(
                query=new_query,
                intent=intent,
                output_sentiment=output_sentiment,
                documents=documents,
                history=history,
                hallucination_retries=hallucination_retries - 1
            )
            
        logger.info("Response generated successfully without hallucinations")
        return response
        
    except Exception as e:
        logger.error(f"Error in rag_response: {str(e)}")
        if hallucination_retries > 1:
            logger.info(f"Attempting retry after error. Retries remaining: {hallucination_retries-1}")
            return rag_response(
                query=query,
                intent=intent,
                output_sentiment=output_sentiment,
                documents=documents,
                history=history,
                hallucination_retries=hallucination_retries - 1
            )
        else:
            logger.error("Maximum retries exceeded after error")
            raise RuntimeError(f"Failed to generate response after maximum retries: {str(e)}")


def rag_agent(
    query: str, 
    user_intent: str, 
    output_emotion: str, 
    history: Optional[str] = None,
    retrieval_retries: int = MAX_RETRIEVAL_RETRIES,
    answer_retries: int = MAX_ANSWER_RETRIES
) -> str:
    """RAG model for response generation with document retrieval and quality checks.

    Args:
        query: The user's query text
        user_intent: The user's intent classification
        output_emotion: The desired output emotion/sentiment
        history: Optional conversation history
        retrieval_retries: Number of remaining retries for document retrieval
        answer_retries: Number of remaining retries for answer quality

    Returns:
        The generated response from the RAG model

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If maximum retries are exceeded or if document retrieval fails
    """
    # Input validation
    if not query or not user_intent or not output_emotion:
        logger.error("Missing required parameters")
        raise ValueError("Query, user_intent, and output_emotion are required")
    
    # Track retries for document retrieval
    if retrieval_retries <= 0:
        logger.warning("Maximum retrieval retries exceeded")
        return "I'm sorry, but I couldn't find relevant information to answer your question accurately."
    
    try:
        # Get documents
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        documents = get_documents(query=query, limit=10)
        
        # Grade documents for relevance
        logger.info("Grading documents for relevance")
        graded_docs = graded_documents(query=query, documents=documents)
        
        # Check if documents are relevant
        if len(graded_docs) == 0:
            logger.warning(f"No relevant documents found. Retries remaining: {retrieval_retries-1}")
            
            # Generate a new query and retry with decremented counter
            reformulated_query = requery(query=query)
            logger.info(f"Reformulated query: {reformulated_query[:50]}...")
            
            return rag_agent(
                query=reformulated_query,
                user_intent=user_intent,
                output_emotion=output_emotion,
                history=history,
                retrieval_retries=retrieval_retries - 1,
                answer_retries=answer_retries
            )
        
        # Generate response
        logger.info("Generating response with relevant documents")
        response = rag_response(
            query=query,
            intent=user_intent,
            output_sentiment=output_emotion,
            documents=graded_docs,
            history=history
        )
        
        # Grade the answer
        logger.info("Grading the generated answer")
        grade = answer_grade(answer=response, question=query)
        
        # Check if the answer is relevant
        if str(grade).lower() == "no" and answer_retries > 0:
            logger.warning(f"Generated answer not relevant. Retries remaining: {answer_retries-1}")
            
            # Generate a new query and retry with decremented counter
            reformulated_query = requery(query=query)
            logger.info(f"Reformulated query for answer: {reformulated_query[:50]}...")
            
            return rag_agent(
                query=reformulated_query,
                user_intent=user_intent,
                output_emotion=output_emotion,
                history=history,
                retrieval_retries=retrieval_retries,
                answer_retries=answer_retries - 1
            )
        
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in rag_agent: {str(e)}")
        if retrieval_retries > 1:
            logger.info(f"Attempting retry after error. Retries remaining: {retrieval_retries-1}")
            return rag_agent(
                query=query,
                user_intent=user_intent,
                output_emotion=output_emotion,
                history=history,
                retrieval_retries=retrieval_retries - 1,
                answer_retries=answer_retries
            )
        else:
            logger.error("Maximum retries exceeded after error")
            raise RuntimeError(f"Failed to generate response after maximum retries: {str(e)}")


def execute_rag_response(
    query: str, 
    user_intent: str, 
    output_emotion: str, 
    history: Optional[str] = None
) -> str:
    """Execute the RAG response with proper error handling and retry management.

    Args:
        query: The user's query text
        user_intent: The user's intent classification
        output_emotion: The desired output emotion/sentiment
        history: Optional conversation history

    Returns:
        The response from the RAG model or an error message
    """
    # Input validation
    if not query:
        logger.error("Empty query provided")
        return "I'm sorry, but I need a question to answer."
    
    try:
        logger.info(f"Starting RAG pipeline for query: {query[:50]}...")
        response = rag_agent(
            query=query, 
            user_intent=user_intent, 
            output_emotion=output_emotion, 
            history=history
        )
        logger.info("RAG pipeline completed successfully")
        return response
        
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        return f"I couldn't process your question due to a validation error: {str(e)}"
        
    except RuntimeError as e:
        logger.error(f"Runtime error in RAG pipeline: {str(e)}")
        return "I'm sorry, but I encountered an issue while processing your question. Please try rephrasing it."
        
    except Exception as e:
        logger.error(f"Unexpected error in RAG pipeline: {str(e)}")
        return "I apologize, but something went wrong while trying to answer your question. Please try again later."