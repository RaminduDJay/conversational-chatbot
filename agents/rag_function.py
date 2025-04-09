import logging
from typing import Optional, List, Tuple, Dict, Any
from functools import lru_cache

from agents.rag_db import get_results
from agents.clients import cohere_client
from agents.rag_agents import grade_answer, grade_document, check_hallucinations, llm_answer, reformulate_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_COLLECTION = "podcasts"
DEFAULT_LIMIT = 5
RERANKER_MODEL = "rerank-english-v3.0"
RERANKER_TOP_N = 5

def get_documents(query: str, limit: Optional[int] = DEFAULT_LIMIT, collection_name: str = DEFAULT_COLLECTION, 
                  use_reranker: bool = False) -> List[str]:
    """
    Retrieve documents from vector database based on the query.

    Args:
        query: The query text to search for
        limit: Maximum number of results to return (default: 5)
        collection_name: Name of the collection to search in (default: "podcasts")
        use_reranker: Whether to use the reranker model (default: False)

    Returns:
        List of retrieved documents

    Raises:
        Exception: If document retrieval fails
    """
    try:
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        limit = limit or DEFAULT_LIMIT
        results = get_results(query, collection_name=collection_name, limit=limit)
        
        if not results:
            logger.info(f"No results found for query: {query}")
            return []

        # Apply reranking if requested
        if use_reranker and cohere_client:
            try:
                logger.info(f"Reranking {len(results)} documents")
                reranker = cohere_client.rerank(
                    model=RERANKER_MODEL, 
                    query=query, 
                    documents=results, 
                    return_documents=False, 
                    top_n=min(RERANKER_TOP_N, len(results))
                )
                indices = [reranker[i].index for i in range(len(reranker))]
                documents = [results[i] for i in indices]
                return documents
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Falling back to original results.")
                return results
        else:
            return results
    except Exception as e:
        logger.error(f"Error in get_documents: {e}")
        raise

def grade_documents(query: str, documents: List[str]) -> List[str]:
    """
    Filter documents based on relevance to the query.

    Args:
        query: The query text
        documents: List of documents to grade

    Returns:
        List of documents that passed the relevance check

    Raises:
        Exception: If document grading fails
    """
    try:
        if not documents:
            logger.warning("No documents provided for grading")
            return []

        final_document_list = []
        
        for i, document in enumerate(documents):
            try:
                logger.info(f"Grading document {i+1}/{len(documents)}")
                grade = grade_document(query=query, document=document)
                binary_score = grade.content[-1].parsed.binary_score
                
                if binary_score == "yes":
                    final_document_list.append(document)
                    logger.info(f"Document {i+1} passed grading")
                else:
                    logger.info(f"Document {i+1} failed grading")
            except Exception as e:
                logger.error(f"Error grading document {i+1}: {e}")
                # Skip this document on error
                continue

        logger.info(f"{len(final_document_list)}/{len(documents)} documents passed grading")
        return final_document_list
    except Exception as e:
        logger.error(f"Error in grade_documents: {e}")
        raise

def llm_generation(query: str, intent: str, output_sentiment: str, documents: List[str], 
                   history: str = "") -> Tuple[str, List[str]]:
    """
    Generate an answer using the LLM model.

    Args:
        query: The query text
        intent: The user intent
        output_sentiment: The desired emotional tone for the output
        documents: List of relevant documents
        history: Conversation history (default: empty string)

    Returns:
        Tuple of (generated answer, documents used)

    Raises:
        Exception: If answer generation fails
    """
    try:
        if not documents:
            logger.warning("No documents provided for LLM generation")
            return "I couldn't find any relevant information to answer your question.", []

        logger.info(f"Generating answer using {len(documents)} documents")
        result = llm_answer(
            query=query, 
            user_intent=intent, 
            output_emotion=output_sentiment, 
            documents=documents, 
            history=history
        )
        
        logger.info("Answer generated successfully")
        return result, documents
    except Exception as e:
        logger.error(f"Error in llm_generation: {e}")
        return f"I'm sorry, I encountered an error while generating a response: {str(e)}", documents

def check_hallucinations(documents: List[str], answer: str) -> str:
    """
    Check if the answer is grounded in the provided documents.

    Args:
        documents: List of reference documents
        answer: The generated answer text

    Returns:
        "yes" if answer is grounded in facts, "no" if hallucinated

    Raises:
        Exception: If hallucination check fails
    """
    try:
        if not documents or not answer:
            logger.warning("Missing documents or answer for hallucination check")
            return "no"

        logger.info("Checking for hallucinations in the generated answer")
        result = check_hallucinations(document=documents, answer=answer)
        binary_score = result.content[-1].parsed.binary_score
        
        logger.info(f"Hallucination check result: {binary_score}")
        return binary_score
    except Exception as e:
        logger.error(f"Error in check_hallucinations: {e}")
        # Default to "no" (assume hallucination) in case of error
        return "no"

def evaluate_answer(answer: str, question: str) -> str:
    """
    Evaluate if the answer adequately addresses the question.

    Args:
        answer: The generated answer text
        question: The original question text

    Returns:
        "yes" if answer resolves the question, "no" otherwise

    Raises:
        Exception: If answer evaluation fails
    """
    try:
        if not answer or not question:
            logger.warning("Missing answer or question for evaluation")
            return "no"

        logger.info("Evaluating if answer resolves the question")
        grade = grade_answer(answer=answer, question=question)
        binary_score = grade.content[-1].parsed.binary_score
        
        logger.info(f"Answer evaluation result: {binary_score}")
        return binary_score
    except Exception as e:
        logger.error(f"Error in evaluate_answer: {e}")
        # Default to "no" in case of error
        return "no"

def reformulate_query(original_query: str) -> str:
    """
    Reformulate the query to improve retrieval results.

    Args:
        original_query: The original query text

    Returns:
        Reformulated query

    Raises:
        Exception: If query reformulation fails
    """
    try:
        if not original_query.strip():
            logger.warning("Empty query provided for reformulation")
            return original_query

        logger.info(f"Reformulating query: {original_query}")
        new_query = reformulate_query(original_query)
        
        logger.info(f"Original query: '{original_query}' -> Reformulated: '{new_query}'")
        return new_query
    except Exception as e:
        logger.error(f"Error in reformulate_query: {e}")
        # Return original query if reformulation fails
        return original_query

@lru_cache(maxsize=32)
def get_and_grade_documents(query: str, limit: Optional[int] = DEFAULT_LIMIT) -> List[str]:
    """
    Retrieve and grade documents in a single function with caching.
    
    Args:
        query: The query text
        limit: Maximum number of documents to retrieve
        
    Returns:
        List of relevant graded documents
    """
    try:
        documents = get_documents(query, limit)
        if documents:
            return grade_documents(query, documents)
        return []
    except Exception as e:
        logger.error(f"Error in get_and_grade_documents: {e}")
        return []

def complete_rag_pipeline(
    query: str,
    intent: str,
    output_sentiment: str,
    history: str = "",
    limit: Optional[int] = DEFAULT_LIMIT,
    use_reranker: bool = False,
    reformulate: bool = True
) -> Dict[str, Any]:
    """
    Execute the complete RAG pipeline from query to answer.
    
    Args:
        query: User's query
        intent: User's intent
        output_sentiment: Desired emotional tone
        history: Conversation history
        limit: Maximum number of documents
        use_reranker: Whether to use reranking
        reformulate: Whether to reformulate the query
        
    Returns:
        Dictionary with results of each stage and final answer
    """
    try:
        pipeline_results = {
            "original_query": query,
            "reformulated_query": None,
            "documents_retrieved": 0,
            "documents_relevant": 0,
            "answer": None,
            "is_hallucination": None,
            "answers_question": None,
            "error": None
        }
        
        # Step 1: Query reformulation (optional)
        working_query = query
        if reformulate:
            working_query = reformulate_query(query)
            pipeline_results["reformulated_query"] = working_query
        
        # Step 2: Document retrieval
        documents = get_documents(working_query, limit, use_reranker=use_reranker)
        pipeline_results["documents_retrieved"] = len(documents)
        
        # Step 3: Document grading
        relevant_docs = grade_documents(working_query, documents)
        pipeline_results["documents_relevant"] = len(relevant_docs)
        
        # Step 4: LLM generation
        if relevant_docs:
            answer, used_docs = llm_generation(
                query=working_query,
                intent=intent,
                output_sentiment=output_sentiment,
                documents=relevant_docs,
                history=history
            )
            pipeline_results["answer"] = answer
            
            # Step 5: Hallucination check
            pipeline_results["is_hallucination"] = check_hallucinations(
                documents=used_docs,
                answer=answer
            )
            
            # Step 6: Answer evaluation
            pipeline_results["answers_question"] = evaluate_answer(
                answer=answer,
                question=working_query
            )
        else:
            pipeline_results["answer"] = "I couldn't find relevant information to answer your question."
            pipeline_results["is_hallucination"] = "no"
            pipeline_results["answers_question"] = "no"
            
        return pipeline_results
    except Exception as e:
        logger.error(f"Error in complete_rag_pipeline: {e}")
        return {
            "original_query": query,
            "error": str(e),
            "answer": "I'm sorry, I encountered an error while processing your question."
        }