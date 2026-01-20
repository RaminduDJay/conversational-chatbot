import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.clients import get_chat_model
from agents.memory_db import SESSION_ID, zep_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "log_dir": "./logdir",
    "model": "gpt-4o-2024-08-06",
    "search_limit": 3,
    "search_mmr_lambda": 0.8,
    "user_id": "user_1"
}

# Validate environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

class ChatbotEntry(BaseModel):
    """Model for chatbot response entry."""
    answer: str = Field(..., description="Answer to user query, including any URL links, timestamps and speaker name if available in history")
    use_rag: bool = Field(..., description="Whether to use RAG model for response generation")
    user_intent: str = Field(..., description="User intent for the query")
    output_emotion: str = Field(..., description="Emotion the generated response should convey")


def load_system_prompt() -> str:
    """Load system prompt from file or return default."""
    try:
        with open("system_prompts/chatbot_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("System prompt file not found, using default prompt")
        return """
        You are a helpful assistant who is a professional podcast host. 
        Your task is to provide a conversational, engaging, and context-aware answer to the query provided, while reflecting the tone and sentiment of the user's input.\n
        Additionally, you will integrate disfluencies, informal language, and overlapping speech from the conversation when necessary, to maintain a natural and coherent podcast-style flow.
        follow the instructions within the <INS> tags.
        <INS>  
        - You are provided with a user query, a history of previous queries and responses, and a summary of the interaction history so far.  
        - Analyze the user query and history to understand user intent and sentiment, generating a response in a conversational, podcast-like tone. 
        - Use the previous responses and the summarised history to provide contextually relevant responses, maintaining the conversational flow. 

        Toxic Speech Handling: \n 
        - If you detect toxic or gibberish speech, acknowledge the sentiment and generate a meaningful response that reflects empathy or understanding. Respond in a way that maintains a conversational, respectful tone, just as a podcast host would manage heated or difficult conversations. For example, you might say: "I can sense there's frustration here, but let's keep this respectful and productive."  
        - In such cases, use_rag=False, user_intent=Toxic, output_emotion=None.  

        Using History:\n  
        - If the answer to the user query is available in the previous responses, generate a response based on the history, including the speaker name, timestamp, and YouTube link in the following format: [(02:16:41)](https://youtube.com/watch?v=tYrdMjVXyNg&t=8201).  
        - use_rag=False for these cases where a complete and concise answer can be derived from the history.  
        - if the answer is available but user explicitly asks to use find more information / explain or if you feel the answer in history is vague or irrelevant then use_rag=True.
        - Maintain the conversational flow of a podcast host, keeping the tone natural and engaging.  

        New Queries:  
        - If the answer is not available in the history, output 'no answer' and enable RAG retrieval with use_rag=True.  
                   
        Conversational Nuances
        - The user may express sarcasm, humor, or frustration. Respond with a light, conversational tone that acknowledges the user's feelings, incorporates humor when appropriate, and gently steers the conversation toward helpful information. 

        The output should contain:  
        - The generated response with speaker name, timestamp, and URL [timestamped YouTube link] (if possible). only provide speaker name, timestamp or URL if available in history else if you do not know, express it,  
        - The use_rag condition (True/False),  
        - The user intent (e.g., Information Request, Toxic),  
        - The output emotion (e.g., Empathy, Neutral, None).  
        </INS>
        """


def search_chatbot_history(query: str = Field(description="Query to search the user long term history")) -> List[str]:
    """Search through chatbot history for previous sessions.
    
    Args:
        query: The query text.
    
    Returns:
        The search results from the chatbot history.
    """
    try:
        data = zep_client.memory.search_sessions(
            session_id=SESSION_ID,
            text=query,
            limit=CONFIG["search_limit"],
            search_scope="messages",
            search_type="mmr",
            mmr_lambda=CONFIG["search_mmr_lambda"],
            user_id=CONFIG["user_id"]
        )
        
        history = []
        for search_results in data.results:
            res = search_results.message.dict()
            history.append(f"content: {res['content']} | role: {res['role']}")
        
        return history
    except Exception as e:
        logger.error(f"Error searching chatbot history: {e}")
        return ["Error retrieving history. Please try again."]


async def chatbot_entry(query: str, history: List[BaseMessage], facts: str) -> ChatbotEntry:
    """Entry point to the chatbot agent.
    
    Args:
        query: User's query
        history: List of previous message exchanges
        facts: Summary of interaction history
    
    Returns:
        ChatbotEntry with response information
    """
    system_prompt = load_system_prompt()
    
    try:
        model = get_chat_model(CONFIG["model"], temperature=0.2).with_structured_output(ChatbotEntry)
        messages = [
            SystemMessage(system_prompt),
            *history,
            HumanMessage(
                "Output answer should contain, the answer to user query, any URL links, "
                "timestamps and the speaker name.\n\n"
                f"Query: {query}\n\nSummarised History:\n{facts}\n"
            ),
        ]
        return await model.ainvoke(messages)
    except Exception as e:
        logger.error(f"Error in chatbot_entry: {e}")
        # Fallback response
        fallback = get_chat_model(CONFIG["model"], temperature=0.2)
        messages = [
            SystemMessage("You are a helpful assistant."),
            HumanMessage("The system encountered an error. Please provide a simple, helpful response."),
        ]
        response = await fallback.ainvoke(messages)
        return ChatbotEntry(
            answer=response.content or "I'm sorry, something went wrong.",
            use_rag=True,
            user_intent="error",
            output_emotion="neutral",
        )
