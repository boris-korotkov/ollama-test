from fastapi import FastAPI
from pydantic import BaseModel
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware

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
    response = llm.invoke(user_input)
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
    uvicorn.run(app, host="127.0.0.1", port=8000)
