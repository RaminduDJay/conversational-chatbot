import logging
import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from zep_cloud import Message
from zep_cloud.client import Zep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate environment variables
ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
if not ZEP_API_KEY:
    logger.error("ZEP_API_KEY environment variable not set")
    raise ValueError("ZEP_API_KEY environment variable not set")

# Initialize Zep client
try:
    zep_client = Zep(api_key=ZEP_API_KEY)
    logger.info("Zep client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Zep client: {e}")
    raise

def generate_session_id() -> str:
    """Generate a unique session ID.
    
    Returns:
        A unique session ID string
    """
    return uuid.uuid4().hex

# Use a function to get session ID rather than a global variable
SESSION_ID = generate_session_id()

def create_user(user_id: str, email: str, firstname: str, lastname: str, metadata: Optional[Dict] = None) -> Dict:
    """Create a new user in Zep.
    
    Args:
        user_id: Unique identifier for the user
        email: User's email address
        firstname: User's first name
        lastname: User's last name
        metadata: Optional metadata dictionary
        
    Returns:
        The created user object
        
    Raises:
        ValueError: If required parameters are missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not user_id or not email:
        logger.error("User ID and email are required")
        raise ValueError("User ID and email are required")
    
    # Set default metadata if none provided
    if metadata is None:
        metadata = {}
    
    try:
        new_user = zep_client.user.add(
            user_id=user_id,
            email=email,
            first_name=firstname,
            last_name=lastname,
            metadata=metadata,
        )
        logger.info(f"User created successfully: {user_id}")
        return new_user
    except Exception as e:
        logger.error(f"Error creating user {user_id}: {e}")
        raise

def create_session(user_id: str, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
    """Create a new session in Zep.
    
    Args:
        user_id: ID of the user for this session
        session_id: Optional session ID (generated if not provided)
        metadata: Optional metadata dictionary
        
    Returns:
        The created session object
        
    Raises:
        ValueError: If required parameters are missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not user_id:
        logger.error("User ID is required")
        raise ValueError("User ID is required")
    
    # Generate session ID if not provided
    if not session_id:
        session_id = generate_session_id()
    
    # Set default metadata if none provided
    if metadata is None:
        metadata = {}
    
    try:
        new_session = zep_client.memory.add_session(
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )
        logger.info(f"Session created successfully: {session_id} for user: {user_id}")
        return new_session
    except Exception as e:
        logger.error(f"Error creating session for user {user_id}: {e}")
        raise

def delete_session(session_id: str) -> bool:
    """Delete a session from Zep.
    
    Args:
        session_id: ID of the session to delete
        
    Returns:
        True if deletion was successful
        
    Raises:
        ValueError: If session_id is missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not session_id:
        logger.error("Session ID is required")
        raise ValueError("Session ID is required")
    
    try:
        zep_client.memory.delete(session_id)
        logger.info(f"Session deleted successfully: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise

def delete_user(user_id: str) -> bool:
    """Delete a user from Zep.
    
    Args:
        user_id: ID of the user to delete
        
    Returns:
        True if deletion was successful
        
    Raises:
        ValueError: If user_id is missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not user_id:
        logger.error("User ID is required")
        raise ValueError("User ID is required")
    
    try:
        zep_client.user.delete(user_id)
        logger.info(f"User deleted successfully: {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise

def add_memory(session_id: str, user_content: str, assistant_content: str, 
               summary_instruction: Optional[str] = None) -> Dict:
    """Add a memory entry to a session.
    
    Args:
        session_id: ID of the session
        user_content: Content from the user
        assistant_content: Content from the assistant
        summary_instruction: Optional instruction for summary generation
        
    Returns:
        The created memory object
        
    Raises:
        ValueError: If required parameters are missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not session_id or not user_content or not assistant_content:
        logger.error("Session ID, user content, and assistant content are required")
        raise ValueError("Session ID, user content, and assistant content are required")
    
    # Default summary instruction if none provided
    if not summary_instruction:
        summary_instruction = "Summarize the conversation highlighting the key points and the query discussed provide URLS, timestamps and youtube urls"
    
    try:
        messages = [
            Message(
                content=user_content,
                role="user",
                role_type="user",
            ),
            Message(
                content=assistant_content,
                role="assistant",
                role_type="assistant",
            )
        ]
        
        memory = zep_client.memory.add(
            session_id=session_id,
            messages=messages,
            summary_instruction=summary_instruction,
        )
        logger.info(f"Memory added successfully to session: {session_id}")
        return memory
    except Exception as e:
        logger.error(f"Error adding memory to session {session_id}: {e}")
        raise

def get_memory(session_id: str) -> Dict:
    """Get memory for a session.
    
    Args:
        session_id: ID of the session
        
    Returns:
        The memory object for the session
        
    Raises:
        ValueError: If session_id is missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not session_id:
        logger.error("Session ID is required")
        raise ValueError("Session ID is required")
    
    try:
        memory = zep_client.memory.get(session_id)
        logger.info(f"Memory retrieved successfully for session: {session_id}")
        return memory
    except Exception as e:
        logger.error(f"Error retrieving memory for session {session_id}: {e}")
        raise

def search_memory(session_id: str, query: str, limit: int = 5) -> List[Dict]:
    """Search memory for relevant messages.
    
    Args:
        session_id: ID of the session to search
        query: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of search results
        
    Raises:
        ValueError: If required parameters are missing
        Exception: If Zep API call fails
    """
    # Validate inputs
    if not session_id or not query:
        logger.error("Session ID and query are required")
        raise ValueError("Session ID and query are required")
    
    try:
        results = zep_client.memory.search(
            session_id=session_id,
            text=query,
            limit=limit
        )
        logger.info(f"Search completed successfully for session: {session_id}")
        return results
    except Exception as e:
        logger.error(f"Error searching memory for session {session_id}: {e}")
        raise

def summarize_conversation(conversation: str) -> str:
    """Summarize a conversation using LLM.
    
    Args:
        conversation: The conversation text to summarize
        
    Returns:
        Summarized text
        
    Raises:
        Exception: If LLM processing fails
    """
    if not conversation:
        logger.warning("Empty conversation provided for summarization")
        return "No conversation to summarize."
    
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        response = model.invoke(
            [
                SystemMessage(
                    "You are an expert transcriber. You will summarise a text containing "
                    "a reply from a podcast host. Your summary must contain what was spoken, "
                    "who spoke about it and the timestamp and url in the format [timestamp](url). "
                    "Provide only the summary and nothing else."
                ),
                HumanMessage(conversation),
            ]
        )
        return response.content
    except Exception as e:
        logger.error(f"Error summarizing conversation: {e}")
        return f"Error summarizing conversation: {str(e)}"
