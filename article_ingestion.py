import os
import requests
import hashlib
# import faiss
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
        # index = FAISS.read_index(VECTOR_DB_FILE)
        index = FAISS.load(VECTOR_DB_FILE)
        # Load URL map
        with open(URL_MAP_FILE, "rb") as f:
            url_map = pickle.load(f)
        return index, url_map
    else:
        # Create a new FAISS index if one doesn't exist
        # index = FAISS.IndexFlatL2(384)  # 384-dimensional embeddings from all-MiniLM-L6-v2
        index = FAISS.new_flat_l2_index(384)  # 384-dimensional embeddings from all-MiniLM-L6-v2
        return index, {}
    


# Save FAISS index
def save_vector_db(index, url_map, vector_db_dir=VECTOR_DB_DIR):
    # Ensure the directory exists
    os.makedirs(vector_db_dir, exist_ok=True)

    # Save FAISS index using FAISS's native save method
    FAISS.write_index(index, VECTOR_DB_FILE)

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

    
