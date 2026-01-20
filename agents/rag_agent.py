import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, validator

from agents.clients import get_chat_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Output formats for agents
class BinaryScoreBase(BaseModel):
    """Base model for binary score responses."""
    binary_score: str = Field(description="Binary response, 'yes' or 'no'")
    
    @validator('binary_score')
    def validate_binary_score(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError("binary_score must be either 'yes' or 'no'")
        return v.lower()

class GradeDocuments(BinaryScoreBase):
    """Binary score for relevance check on retrieved documents."""
    pass

class GradeHallucinations(BinaryScoreBase):
    """Binary score for hallucination present in generation answer."""
    pass

class GradeAnswer(BinaryScoreBase):
    """Binary score to assess answer addresses question."""
    pass

class RequeryOutput(BaseModel):
    """Structure for requery output"""
    new_query: str = Field(description="Reformulated query for better retrieval")

# Base agent class to reduce code duplication
class GradingAgent:
    """Base class for grading agents to reduce code duplication."""
    
    @staticmethod
    def format_documents(documents: List[str]) -> str:
        """Format a list of documents into a single string."""
        return "\n".join(documents)

def grade_document(query: str, document: str) -> GradeDocuments:
    """
    Document grading agent, assesses if a document is relevant to a query.
    
    Args:
        query: The user's question
        document: The retrieved document to evaluate
        
    Returns:
        GradeDocuments: Binary score indicating relevance
    """
    try:
        model = get_chat_model("gpt-4o-mini", temperature=0.0).with_structured_output(GradeDocuments)
        messages = [
            SystemMessage("""You are a grader assessing relevance of a retrieved document to a user question.
                    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
                    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
                    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                    """),
            HumanMessage(f"Query: {query}\n\nDocument: {document}"),
        ]
        return model.invoke(messages)
    except Exception as e:
        logger.error(f"Error in grade_document: {e}")
        # Default to including document in case of error (safer approach)
        return GradeDocuments(binary_score="yes")

def check_hallucinations(document: List[str], answer: str) -> GradeHallucinations:
    """
    Hallucination grading agent, checks if an answer is grounded in provided documents.
    
    Args:
        document: List of relevant documents
        answer: The generated answer to evaluate
        
    Returns:
        GradeHallucinations: Binary score indicating factual grounding
    """
    try:
        formatted_document = GradingAgent.format_documents(document)
        model = get_chat_model("gpt-4o-mini", temperature=0.0).with_structured_output(GradeHallucinations)
        messages = [
            SystemMessage("""You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
                Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts else 'no'.
                The response can be 'yes' or 'no' and nothing else.
                    """),
            HumanMessage(f"Set of facts:\n\n{formatted_document}\n\nLLM generation: {answer}"),
        ]
        return model.invoke(messages)
    except Exception as e:
        logger.error(f"Error in check_hallucinations: {e}")
        # Default to assuming hallucination in case of error (safer approach)
        return GradeHallucinations(binary_score="no")

def grade_answer(answer: str, question: str) -> GradeAnswer:
    """
    Answer grading agent, evaluates if an answer addresses the original question.
    
    Args:
        answer: The generated answer to evaluate
        question: The original user question
        
    Returns:
        GradeAnswer: Binary score indicating question resolution
    """
    try:
        model = get_chat_model("gpt-4o-mini", temperature=0.0).with_structured_output(GradeAnswer)
        messages = [
            SystemMessage("""You are a grader assessing whether an answer addresses / resolves a question.
                You do not need to be overly strict. The goal is to filter out if irrelevant answers created.
                As long as the answer is relevant to the question, grade it as relevant.
                Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question else 'no'.
                    """),
            HumanMessage(f"Question: {question}\n\nAnswer: {answer}"),
        ]
        return model.invoke(messages)
    except Exception as e:
        logger.error(f"Error in grade_answer: {e}")
        # Default to assuming answer is not relevant in case of error
        return GradeAnswer(binary_score="no")

def llm_answer(
    query: str, 
    user_intent: str, 
    output_emotion: str, 
    documents: List[str], 
    history: Optional[str] = ""
) -> str:
    """
    Generate a podcast host-styled answer using retrieved documents and conversation context.

    Args:
        query: The user's question
        user_intent: The inferred user intent
        output_emotion: The desired emotional tone for the response
        documents: The list of relevant documents
        history: The conversation history (optional)

    Returns:
        The generated podcast host-styled answer
    """
    try:
        formatted_documents = GradingAgent.format_documents(documents)
        
        # Split system prompt into more manageable components
        system_base = """You are a state-of-the-art Q&A chatbot designed to respond in the persona of a podcast host."""
        
        system_task = """Your task is to provide a conversational, engaging, and context-aware answer 
        to the query provided, while reflecting the tone and sentiment of the user's input."""
        
        system_style = """Integrate disfluencies, informal language, and overlapping speech from the conversation 
        when necessary, to maintain a natural and coherent podcast-style flow."""
        
        system_answer_dev = """
        Answer Development:
        - Read the provided query, user intent, output emotion, and retrieved documents.
        - Analyze the sentiment and tone of the query.
        - Formulate a response based on the query and provided documents.
        - Refine your response to match the appropriate tone.
        - Consider the conversation history for context.
        - Use a conversational, flowing style with natural pauses and transitions.
        """
        
        system_tone = """
        Sentiment and Tone Adjustment:
        - For negative sentiment: respond with empathy and calm.
        - For humorous/sarcastic queries: mirror with wit and lightheartedness.
        - For neutral queries: maintain a balanced, informative tone.
        """
        
        system_attribution = """
        Source Attribution:
        - Your retrieved data consists of transcript URLs.
        - Include YouTube video links and timestamps related to your answer sources.
        - Format: [(time)](youtube url)
        """
        
        system_reminder = """
        Remember:
        - Stay engaging and natural as a podcast host.
        - Handle conversational elements naturally.
        - Always provide proper source attribution.
        """
        
        # Combine system prompts
        system_prompt = "\n\n".join([
            system_base, 
            system_task, 
            system_style, 
            system_answer_dev,
            system_tone,
            system_attribution,
            system_reminder
        ])
        
        model = get_chat_model("gpt-4o-2024-08-06", temperature=0.2)
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(
                f"Documents:\n{formatted_documents}\n\n"
                f"Conversation history: {history}\n\n"
                f"Query: {query}\nUser Intent: {user_intent}\n"
                f"Answer with output emotion: {output_emotion}"
            ),
        ]
        return model.invoke(messages).content
    except Exception as e:
        logger.error(f"Error in llm_answer: {e}")
        return "I apologize, but I'm having trouble generating a response. Could you please try again with your question?"

def requery(query: str) -> RequeryOutput:
    """
    Reformulate a user query to improve document retrieval results.

    Args:
        query: The original user query

    Returns:
        RequeryOutput: Object containing the reformulated query
    """
    try:
        model = get_chat_model("gpt-4o-mini", temperature=0.0).with_structured_output(RequeryOutput)
        messages = [
            SystemMessage("""You are a query reformulation expert. Given a user query, you must rewrite it 
            to improve document retrieval results. Your output should contain only the new query.
            Focus on:
            - Clarifying ambiguous terms
            - Expanding abbreviations
            - Adding relevant synonyms
            - Removing unnecessary words
            - Maintaining the core intent
            """),
            HumanMessage(f"Original query: {query}"),
        ]
        return model.invoke(messages)
    except Exception as e:
        logger.error(f"Error in requery: {e}")
        # Return original query if reformulation fails
        return RequeryOutput(new_query=query)
