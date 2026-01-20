import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration with defaults
class Config:
    """Configuration settings with environment variable fallbacks."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "https://3973cdf9-4ba6-40b1-ae92-b2f952f82fb9.europe-west3-0.gcp.cloud.qdrant.io:6333")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))
    ENTITY_MODEL = os.getenv("ENTITY_MODEL", "gpt-4o-mini")

# Validate required environment variables
def validate_config() -> None:
    """Validate that all required environment variables are set."""
    missing_vars = []
    
    if not Config.OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not Config.QDRANT_CLOUD_API_KEY:
        missing_vars.append("QDRANT_CLOUD_API_KEY")
        
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients with proper error handling
def init_clients():
    """Initialize API clients with proper error handling."""
    try:
        validate_config()
        
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_CLOUD_API_KEY,
            timeout=10.0  # Add timeout for operations
        )
        embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        entity_llm = ChatOpenAI(model=Config.ENTITY_MODEL, temperature=0.0)

        logger.info("All clients initialized successfully")
        return embeddings, qdrant_client, entity_llm
    
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

# Initialize clients
try:
    embeddings_client, qdrant_client, entity_llm = init_clients()
except Exception as e:
    logger.error(f"Client initialization failed: {e}")
    raise

# Constants
KEYWORD_PROMPT = """
Your task is to analyze the query and identify the entities in the query.
The output must contain only the entities separated by comma and no other details. 
Do not share anything other than what you are asked to.
You must strictly follow the instruction.
Only provide the keywords found and nothing else.
"""

@lru_cache(maxsize=128)
def get_embedding(text: str) -> List[float]:
    """
    Get OpenAI embedding for the given text with caching.
    
    Args:
        text: The text to embed
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If there's an error getting embeddings
    """
    try:
        return embeddings_client.embed_query(text)
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

def get_entities(text: str) -> List[str]:
    """
    Get entities from the given text using OpenAI.
    
    Args:
        text: The text to extract entities from
        
    Returns:
        List of entities found in the text
        
    Raises:
        Exception: If there's an error extracting entities
    """
    try:
        response = entity_llm.invoke(
            [
                SystemMessage(KEYWORD_PROMPT),
                HumanMessage(text),
            ]
        )
        entities = [entity.strip() for entity in response.content.split(",")]
        return [entity for entity in entities if entity]  # Filter out empty strings
    
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        # Return a reasonable fallback - just the original query words
        return text.split()

def hybrid_search(
    collection_name: str,
    query: str,
    limit: int = 5,
    subtopic: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    title: Optional[str] = None,
    full_text_search: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for similar documents using a hybrid vector and keyword approach.
    
    Args:
        collection_name: The Qdrant collection name
        query: The search query text
        limit: Maximum number of results to return
        subtopic: Optional filter by subtopic
        speakers: Optional filter by speakers
        title: Optional filter by title
        full_text_search: Whether to include keyword matching
        
    Returns:
        List of matching documents with metadata
        
    Raises:
        Exception: If there's an error during search
    """
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Build filter conditions
        must_conditions = []
        should_conditions = []
        
        # Add metadata filters
        if subtopic:
            must_conditions.append(models.FieldCondition(
                key="subtopic", 
                match=models.MatchValue(value=subtopic)
            ))
            
        if speakers:
            must_conditions.append(models.FieldCondition(
                key="metadata.speakers", 
                match=models.MatchAny(any=speakers)
            ))
            
        if title:
            must_conditions.append(models.FieldCondition(
                key="metadata.title", 
                match=models.MatchValue(value=title)
            ))
        
        # Add text search conditions if enabled
        if full_text_search:
            entities = get_entities(query)
            for word in entities:
                should_conditions.append(models.FieldCondition(
                    key="content", 
                    match=models.MatchText(text=word)
                ))
        
        # Perform search - more efficient single search with combined filters
        search_filter = models.Filter(must=must_conditions)
        
        # Add should conditions if we have them
        if should_conditions and full_text_search:
            search_filter.should = should_conditions
        
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit * 2,  # Get more results to account for potential duplicates
            with_payload=True,
            score_threshold=0.0
        )
        
        # Process results
        retrieved_docs = process_search_results(search_results)
        
        # Return top results up to limit
        return retrieved_docs[:limit]
    
    except UnexpectedResponse as e:
        logger.error(f"Qdrant error: {e}")
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise

def process_search_results(results) -> List[Dict[str, Any]]:
    """
    Process search results to extract relevant fields and remove duplicates.
    
    Args:
        results: Raw search results from Qdrant
        
    Returns:
        Processed and deduplicated results
    """
    # Extract data and deduplicate in a single pass
    seen_ids: Set[str] = set()
    processed_results = []
    
    for hit in results:
        doc_id = hit.id
        
        # Skip if already seen
        if doc_id in seen_ids:
            continue
            
        seen_ids.add(doc_id)
        
        # Extract fields with safe defaults
        processed_results.append({
            "id": doc_id,
            "subtopic": hit.payload.get("subtopic", ""),
            "speakers": hit.payload.get("speakers", []),
            "content": hit.payload.get("content", ""),
            "title": hit.payload.get("title", ""),
            "url": hit.payload.get("url", ""),
            "timestamp": hit.payload.get("timestamp", ""),
            "score": hit.score
        })
    
    # Sort by score
    return sorted(processed_results, key=lambda x: x["score"], reverse=True)

def markdown_template(data: Dict[str, Any]) -> str:
    """
    Format search result as markdown.
    
    Args:
        data: Document data dictionary
        
    Returns:
        Formatted markdown string
    """
    # Handle potentially missing fields
    title = data.get("title", "Untitled")
    subtopic = data.get("subtopic", "N/A")
    speakers = data.get("speakers", [])
    speakers_str = ", ".join(speakers) if speakers else "Unknown"
    timestamp = data.get("timestamp", "N/A")
    url = data.get("url", "#")
    content = data.get("content", "No content available")
    
    return f"""
    **Document**:
    - **Title**: {title}
    - **Subtopic**: {subtopic}
    - **Speakers**: {speakers_str}
    - **Timestamp**: {timestamp}
    - **URL**: [{url}]

    **Content**:
    {content}
    """

def get_results(
    query: str, 
    collection_name: str, 
    limit: int = 5,
    subtopic: Optional[str] = None,
    speakers: Optional[List[str]] = None,
    title: Optional[str] = None
) -> List[str]:
    """
    Get formatted search results for the given query.
    
    Args:
        query: The search query
        collection_name: The Qdrant collection name
        limit: Maximum number of results
        subtopic: Optional filter by subtopic
        speakers: Optional filter by speakers
        title: Optional filter by title
        
    Returns:
        List of formatted markdown results
        
    Raises:
        Exception: If search fails
    """
    try:
        results = hybrid_search(
            collection_name=collection_name, 
            query=query, 
            limit=limit,
            subtopic=subtopic,
            speakers=speakers,
            title=title
        )
        return [markdown_template(data) for data in results]
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        return [f"Error retrieving results: {str(e)}"]
