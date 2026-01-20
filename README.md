# conversational-chatbot

Podcast chatbot for chat based on transcripts.


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file
```bash
QDRANT_CLOUD_API_KEY
OPENAI_API_KEY
ZEP_API_KEY
```
Where to get API keys\
`QDRANT_CLOUD_API_KEY` from https://qdrant.tech \
`OPENAI_API_KEY` from https://platform.openai.com \
`ZEP_API_KEY` from https://www.getzep.com \


## Run Locally

Clone the project

```bash
  git clone https://github.com/RaminduDJay/conversational-chatbot.git
```

Go to the project directory

```bash
  cd conversational-chatbot
```

Install dependencies
if uv package manager is not installed, install it.

```bash
  pip install uv 
```

install dependencies on to virtual environment

```bash
  uv run app.py
```

run data pipeline  \
go to notebooks folder and run `data_injest.ipynb` to populate database and setup user \
ONLY IF .env not shared \

run app 

```bash
  uv run streamlit run app.py
```

This project uses OpenAI + LangChain for LLM orchestration.
