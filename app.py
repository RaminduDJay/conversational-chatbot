import streamlit as st
import asyncio
from typing import List, Optional
import logging
from contextlib import contextmanager

from agents.entry import chatbot_entry
from agents.memory_db import (
    get_memory, 
    delete_session, 
    add_memory, 
    create_session, 
    summarize_conversation, 
    SESSION_ID
)
from agents.podcast_agent import execute_rag_response
from ell import Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY_SIZE = 100
DISPLAY_MESSAGE_COUNT = 20
DEFAULT_USER = "user_1"

# Page configuration
st.set_page_config(
    page_title="Podcast Assistant",
    page_icon='🎙️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("Podcast Chat Assistant")

# Initialize session state
if "session_active" not in st.session_state:
    st.session_state.session_active = False
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "history" not in st.session_state:
    st.session_state.history = []

@contextmanager
def error_handler(operation_name: str):
    """Context manager for handling errors with proper logging and user feedback."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation_name}: {str(e)}")
        st.error(f"Something went wrong during {operation_name}. Please try again.")
        
def handle_create_session():
    """Create a new chat session."""
    with error_handler("session creation"):
        create_session(user_id=DEFAULT_USER, session_id=SESSION_ID)
        st.session_state.session_active = True
        st.session_state.history = []
        st.session_state.messages = [
            Message(role="assistant", content="Hello! I'm a podcast host. Ask me anything about the podcast.")
        ]
        logger.info(f"Created new session with ID: {SESSION_ID}")

def handle_delete_session():
    """Delete the current chat session."""
    with error_handler("session deletion"):
        delete_session(SESSION_ID)
        st.session_state.session_active = False
        st.session_state.history = []
        st.session_state.messages = []
        logger.info(f"Deleted session with ID: {SESSION_ID}")

def get_facts_content() -> str:
    """Retrieve facts from memory and format them."""
    try:
        facts = get_memory(session_id=SESSION_ID)
        if facts and hasattr(facts, 'facts') and facts.facts:
            return "\n".join(facts.facts)
        return ""
    except Exception as e:
        logger.error(f"Error retrieving facts: {str(e)}")
        return ""

def prune_history():
    """Prevent history from growing too large."""
    if len(st.session_state.history) > MAX_HISTORY_SIZE:
        st.session_state.history = st.session_state.history[-MAX_HISTORY_SIZE:]
        logger.info(f"Pruned history to {MAX_HISTORY_SIZE} items")

def get_message_text(message) -> str:
    """Extract text content from a message object safely."""
    if isinstance(message.content, str):
        return message.content
    elif hasattr(message.content, "__getitem__") and len(message.content) > 0:
        if hasattr(message.content[-1], "text"):
            return message.content[-1].text
    return str(message.content)

# UI components
with st.expander("Session Management"):
    col1, col2 = st.columns(2)
    with col1:
        st.button("Create New Session", on_click=handle_create_session)
    with col2:
        st.button("Clear Current Session", on_click=handle_delete_session)

async def process_user_input(prompt: str, facts_content: str):
    """Process user input and generate response."""
    # Calculate which portion of history to use for context
    if len(st.session_state.messages) > DISPLAY_MESSAGE_COUNT:
        combined_history = st.session_state.history[:-DISPLAY_MESSAGE_COUNT] + st.session_state.messages[-DISPLAY_MESSAGE_COUNT:]
    else:
        combined_history = st.session_state.messages
    
    # Get assistant response
    with st.spinner("Thinking..."):
        try:
            response = await chatbot_entry(
                query=prompt,
                history=combined_history,
                facts=facts_content,
            )
            
            # Check if RAG processing is needed
            use_rag = False
            if hasattr(response.content[-1], "parsed") and hasattr(response.content[-1].parsed, "use_rag"):
                use_rag = str(response.content[-1].parsed.use_rag).lower() == "true"
            
            # Handle RAG response if requested
            if use_rag:
                user_intent = response.content[-1].parsed.user_intent if hasattr(response.content[-1].parsed, "user_intent") else ""
                output_emotion = response.content[-1].parsed.output_emotion if hasattr(response.content[-1].parsed, "output_emotion") else ""
                
                rag_response = await execute_rag_response(
                    query=prompt,
                    user_intent=user_intent,
                    output_emotion=output_emotion,
                    history=facts_content,
                )
                assistant_content = rag_response
            else:
                # Use regular response
                assistant_content = response.content[-1].parsed.answer if hasattr(response.content[-1].parsed, "answer") else str(response.content[-1])
            
            # Store message in memory
            await add_memory(
                session_id=SESSION_ID,
                user_content=prompt,
                assistent_content=assistant_content,
            )
            
            # Add to session messages
            st.session_state.messages.append(Message(role="assistant", content=assistant_content))
            
            # Summarize for history
            summary = await summarize_conversation(conversation=assistant_content)
            st.session_state.history.append(Message(role="assistant", content=summary))
            
            return assistant_content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_msg = "I'm sorry, I encountered an error processing your request. Please try again."
            st.session_state.messages.append(Message(role="assistant", content=error_msg))
            return error_msg

async def main():
    """Main application logic."""
    # Display session warning if no active session
    if not st.session_state.session_active:
        st.warning("Please create a session to start chatting")
        return
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.markdown(get_message_text(message))
    
    # Handle user input
    prompt = st.chat_input(placeholder="Ask me anything about the podcast")
    if prompt:
        # Add user message to state
        st.session_state.messages.append(Message(role="user", content=prompt))
        st.session_state.history.append(Message(role="user", content=prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get facts to provide context
        facts_content = get_facts_content()
        
        # Process input and display response
        with st.chat_message("assistant"):
            response = await process_user_input(prompt, facts_content)
            st.markdown(response)
        
        # Maintain reasonable history size
        prune_history()

if __name__ == "__main__":
    asyncio.run(main())