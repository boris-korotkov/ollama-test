import os
import requests
import hashlib
import faiss
import pickle
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

# File paths
LINKS_FILE = "links.txt"
PROCESSED_LINKS_FILE = "processed_links.txt"
VECTOR_DB_FILE = "vector_db.pkl"
VECTOR_DB_DIR = "vector_store"  # directory where your index and url_map files are stored
VECTOR_DB_FILE = os.path.join(VECTOR_DB_DIR, "index.faiss")
URL_MAP_FILE = os.path.join(VECTOR_DB_DIR, "url_map.pkl")

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to fetch and parse article content
def fetch_article(url, retries=3, delay=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            return "\n".join(paragraphs) if paragraphs else None
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    
    print(f"Failed to fetch: {url}")
    return None

# Load processed links
def load_processed_links():
    if os.path.exists(PROCESSED_LINKS_FILE):
        with open(PROCESSED_LINKS_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()

# Save processed links
def save_processed_link(url):
    with open(PROCESSED_LINKS_FILE, "a") as f:
        f.write(url + "\n")

# Load or initialize FAISS index
def load_vector_db():
    if os.path.exists(VECTOR_DB_FILE):
        # Load FAISS index
        index = faiss.read_index(VECTOR_DB_FILE)
        # Load URL map
        with open(URL_MAP_FILE, "rb") as f:
            url_map = pickle.load(f)
        return index, url_map
    else:
        # Create a new FAISS index if one doesn't exist
        index = faiss.IndexFlatL2(384)  # 384-dimensional embeddings from all-MiniLM-L6-v2
        return index, {}
    
# def load_vector_db():
#     if os.path.exists(VECTOR_DB_FILE):
#         # Load the FAISS index using faiss.read_index()
#         index = faiss.read_index(VECTOR_DB_FILE)

#         # Load the URL map using pickle
#         if os.path.exists(URL_MAP_FILE):
#             with open(URL_MAP_FILE, "rb") as f:
#                 url_map = pickle.load(f)
#         else:
#             url_map = {}

#         return index, url_map
#     else:
#         # If the FAISS index file does not exist, create a new one (empty)
#         index = faiss.IndexFlatL2(384)  # 384-dimensional embeddings (adjust if needed)
#         return index, {}
    
# def load_vector_db():
#     if os.path.exists(VECTOR_DB_FILE):
#         with open(VECTOR_DB_FILE, "rb") as f:
#             index, url_map = pickle.load(f)
#         return index, url_map
#     else:
#         index = faiss.IndexFlatL2(384)  # 384-dimensional embeddings
#         return index, {}

# Save FAISS index
# def save_vector_db(index, url_map):
#     with open(VECTOR_DB_FILE, "wb") as f:
#         pickle.dump((index, url_map), f)
def save_vector_db(index, url_map, vector_db_dir=VECTOR_DB_DIR):
    # Ensure the directory exists
    os.makedirs(vector_db_dir, exist_ok=True)

    # Save FAISS index using FAISS's native save method
    faiss.write_index(index, VECTOR_DB_FILE)

    # Optionally, save the url_map using pickle
    with open(URL_MAP_FILE, "wb") as f:
        pickle.dump(url_map, f)


# Main function to process new articles
def process_links():
    processed_links = load_processed_links()
    index, url_map = load_vector_db()

    if not os.path.exists(LINKS_FILE):
        print("No links file found.")
        return

    with open(LINKS_FILE, "r") as f:
        new_links = set(f.read().splitlines()) - processed_links

    if not new_links:
        print("No new links to process.")
        return

    for url in new_links:
        content = fetch_article(url)
        if content:
            embedding = model.encode(content, convert_to_numpy=True)
            index.add(np.array([embedding], dtype=np.float32))  # Add to FAISS index
            url_map[len(url_map)] = url  # Track URLs
            save_processed_link(url)
            print(f"Processed: {url}")
            print("Current URL Map:", url_map)

    save_vector_db(index, url_map)
    print("Database updated successfully.")

if __name__ == "__main__":
    process_links()

    # Debugging: Print stored documents
    
    # # Load embeddings (must match what was used in ingestion)
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # # Load existing FAISS index
    # # vector_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    # # vector_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    # index, url_map = load_vector_db()
    # vector_db = FAISS(index=index)

    # # Retrieve stored documents (testing query)
    # query = "Sun Life total assets under management"
    # retrieved_docs = vector_db.similarity_search(query,embeddings)

    # # Print retrieved documents
    # print("\n--- Retrieved Documents ---")
    # for doc in retrieved_docs:
    #     print(doc.page_content)

    # Now to initialize the FAISS vector store correctly:
    # In the backend.py or any other script, you need to initialize the FAISS index with the required parameters
    class Document:
        def __init__(self, page_content):
            self.page_content = page_content

        def __repr__(self):
            return f"Document(page_content={self.page_content})"
        
    index, url_map = load_vector_db()
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # docstore = InMemoryDocstore(url_map)  # Use the URL map as a simple docstore
    # docstore = InMemoryDocstore({i: {"page_content": url} for i, url in url_map.items()})  # Use the URL map as a simple docstore
    # docstore = InMemoryDocstore({i: Document(url) for i, url in url_map.items()})
    docstore = InMemoryDocstore({i: Document(url) for i, url in url_map.items()})

    # Correcting the index_to_docstore_id mapping
    index_to_docstore_id = {i: i for i in range(len(url_map))}  # Map FAISS index IDs to URL map keys

    # Initialize the FAISS vector store
    vector_db = FAISS(
        index=index,
        embedding_function=embedding_function,
        docstore=docstore,
        # index_to_docstore_id={i: i for i in range(len(url_map))}  # Map index IDs to the URL map keys
        index_to_docstore_id=index_to_docstore_id  # Correct mapping of FAISS index IDs
    )

    # Example usage of the vector store:
    query = "What are the latest updates on the Sun Life financials?"
    results = vector_db.similarity_search(query, k=2)  # Retrieve top 2 relevant documents
    print(results)
