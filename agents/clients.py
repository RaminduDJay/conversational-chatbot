from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

@lru_cache(maxsize=8)
def get_chat_model(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Return a cached OpenAI chat model."""
    return ChatOpenAI(model=model, temperature=temperature)

@lru_cache(maxsize=2)
def get_embeddings(model: str) -> OpenAIEmbeddings:
    """Return a cached OpenAI embeddings client."""
    return OpenAIEmbeddings(model=model)
