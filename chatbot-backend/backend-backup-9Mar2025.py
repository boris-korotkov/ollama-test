from fastapi import FastAPI
from pydantic import BaseModel
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware
# import faiss
from langchain_community.vectorstores import FAISS
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DB_FILE = "vector_db.pkl"
VECTOR_DB_DIR = "vector_store"  # directory where your index and url_map files are stored
VECTOR_DB_FILE = os.path.join(VECTOR_DB_DIR, "index.faiss")
URL_MAP_FILE = os.path.join(VECTOR_DB_DIR, "url_map.pkl")

# Load vector database
def load_vector_db():
    if os.path.exists(VECTOR_DB_FILE):
        # Load FAISS index
        # index = FAISS.read_index(VECTOR_DB_FILE)
        index = FAISS.load(VECTOR_DB_FILE)
        # Load URL map
        with open(URL_MAP_FILE, "rb") as f:
            url_map = pickle.load(f)
        return index, url_map
    else:
        # Create a new FAISS index if one doesn't exist
        # index = FAISS.IndexFlatL2(384)  # 384-dimensional embeddings from all-MiniLM-L6-v2
        index = FAISS.new_flat_l2(384)  # 384-dimensional embeddings from all-MiniLM-L6-v2
        return index, {}



# Search FAISS for relevant articles
def retrieve_relevant_content(query, top_k=2):
    
    if index is None or index.ntotal == 0:
        return ""  # No stored knowledge
    
    query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_texts = []
    for i in indices[0]:
        if i in url_map:
            retrieved_texts.append(f"- {url_map[i]}")  # Returning URL instead of full content
    
    return "\n".join(retrieved_texts) if retrieved_texts else ""

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
llm = OllamaLLM(model="deepseek-r1:8b",verbose=False)

# Define request structure
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    # Retrieve relevant articles
    related_articles = retrieve_relevant_content(user_input)
    # Construct prompt
    prompt = f"User asked: {user_input}\n\nRelevant resources:\n{related_articles}\n\nAnswer concisely:"

    response = llm.invoke(prompt)
    # response = llm.invoke(user_input, max_tokens=100)  # Limit response to 100 tokens
    # response = llm.invoke(user_input, stop_sequence=["\n", "END"]) # Limit response by setting a stop sequence. This tells the model to stop generating further text once it encounters certain tokens.

    # Clean the response by removing <think> and </think> tags
    clean_response = response.replace("<think>", "").replace("</think>", "")
    
    return {"response": clean_response}

# def chat(request: ChatRequest):
#     response = llm.invoke(request.message)
#     return {"response": response}



# Run API
if __name__ == "__main__":
    import uvicorn
    index, url_map = load_vector_db()
    print (index)
    print (url_map)
    uvicorn.run(app, host="127.0.0.1", port=8000)
