import os
import hashlib
from typing import List
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
# from chromadb import Client
# from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb


# Ensure the chroma_db directory exists
os.makedirs("./chroma_db", exist_ok=True)


# Initialize Chroma database
print("Initializing Chroma database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    "articles",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)
print("Chroma database initialized.")

# File to track processed links
PROCESSED_LINKS_FILE = "processed_links.txt"

def read_links_from_file(file_path: str) -> List[str]:
    """Read links from the text file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]

def write_processed_link(link: str):
    """Write a processed link to the tracking file."""
    with open(PROCESSED_LINKS_FILE, "a") as file:
        file.write(link + "\n")

def load_processed_links() -> set:
    """Load already processed links."""
    if not os.path.exists(PROCESSED_LINKS_FILE):
        return set()
    with open(PROCESSED_LINKS_FILE, "r") as file:
        return set(line.strip() for line in file)

def fetch_article_content(url: str) -> str:
    """Fetch and clean article content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract article content (adjust tags as needed)
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text() for p in paragraphs)
        return content.strip()
    except Exception as e:
        print(f"Failed to fetch content from {url}: {e}")
        return ""

def store_article_in_vector_db(url: str, content: str):
    """Store the article's content in the vector database."""
    doc_id = hashlib.sha256(url.encode()).hexdigest()
    collection.upsert(
        documents=[content],
        metadatas=[{"url": url}],
        ids=[doc_id]
    )
    print(f"Stored document with ID: {doc_id} and URL: {url}")
    print(f"Document content length: {len(content)} characters")  # Debugging line

def main():
    """Main function to parse and store articles."""
    link_file = "links.txt"  # Replace with your link file path
    links = read_links_from_file(link_file)
    processed_links = load_processed_links()

    for link in links[:1000]:  # Limit to 1000 daily
        if link in processed_links:
            continue
        content = fetch_article_content(link)
        if content:
            print(f"Fetched content for URL: {link}")
            store_article_in_vector_db(link, content)
            write_processed_link(link)

def test_indexing():
    # Load the database
    print("Loading Chroma database...")
    db = chromadb.PersistentClient(path="./chroma_db")  # Use PersistentClient
    collection = db.get_collection("articles")
    print("Chroma database loaded.")

    # Fetch stored documents
    results = collection.get()

    print(f"Total Indexed Documents: {len(results['documents'])}")
    for i, doc in enumerate(results['documents'][:5]):  # Print first 5 docs
        print(f"\nDocument {i+1}:")
        print(doc[:500])  # Print first 500 characters
      
if __name__ == "__main__":
    main()
    #test_indexing()