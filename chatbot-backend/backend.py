from fastapi import FastAPI
from pydantic import BaseModel
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import os

# Initialize FastAPI app
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama with a model
# llm = Ollama(model="deepseek-r1:8b")  # Change to your preferred model
llm = OllamaLLM(model="deepseek-r1:8b", verbose=False)

# Load the database
print("Loading Chroma database...")
#db = chromadb.PersistentClient(path="./chroma_db")  # Use PersistentClient
# Get the absolute path of the ChromaDB folder one level above
chroma_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "chroma_db"))

# Initialize ChromaDB with the correct path
db = chromadb.PersistentClient(path=chroma_db_path)

collection = db.get_collection("articles")
print("Chroma database loaded.")

# Define request structure
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message

    # Retrieve relevant context from the vector database
    query_results = collection.query(
        query_texts=[user_input],
        n_results=3
    )

    # Combine relevant content for the response
    #retrieved_docs = query_results.get("documents", [])
    retrieved_docs = [doc[0] for doc in query_results.get("documents", []) if doc]

    if not retrieved_docs:
        context = "No relevant documents found. Answer based on general knowledge."
    else:
        context = "\n".join(retrieved_docs)

    # Pass the context to the LLM
    response = llm.invoke(f"{context}\n{user_input}")

    # Clean the response by removing <think> and </think> tags
    clean_response = response.replace("<think>", "").replace("</think>", "")

    return {"response": clean_response}

# Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
