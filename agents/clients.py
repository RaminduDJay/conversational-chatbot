import os
import groq 
import cohere
from dotenv import load_dotenv

load_dotenv()

# Replace OPENAI_API_KEY with DEEPEEK_API_KEY
os.environ["DEEPEEK_API_KEY"] = os.getenv("DEEPEEK_API_KEY")
openai.base_url = "https://api.deepseek.com/v1"

groq_client = groq.Groq(
    api_key=os.getenv('GROQ_API_KEY'),
)

cohere_client = cohere.ClientV2(os.getenv("COHERE_API_KEY"))