import streamlit as st
import os
import sys
import traceback
import requests

# SQLite Patch
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import torch
import openai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# OpenAI API Key Configuration
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Add an input for the API key in the UI if not already set
if not openai.api_key:
    st.sidebar.warning("OpenAI API Key is required!")
    api_key_input = st.sidebar.text_input(
        "Enter your OpenAI API Key", 
        type="password"
    )
    
    if api_key_input:
        openai.api_key = api_key_input
        st.sidebar.success("API Key set successfully!")

# GitHub Authentication
hf_token = st.text_input("Enter your Hugging Face API token:", type="password")

# Store the token in the session state
if hf_token:
    st.session_state.hf_token = hf_token
    st.success("Hugging Face token stored successfully!")

# Use the token in your app
if 'hf_token' in st.session_state:
    st.write("Token is available for use in the app")
else:
    st.warning("Please enter your Hugging Face token")

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize ChromaDB
@st.cache_resource
def get_chroma_client():
    # Use persistent storage to prevent collection conflicts
    return chromadb.PersistentClient(
        path=".chromadb",  # Specify a persistent storage path
        settings=Settings(
            anonymized_telemetry=False,  # Disable telemetry if needed
            allow_reset=True  # Allow resetting the database
        )
    )
def get_or_create_collection(client, collection_name):
    """
    Safely get or create a ChromaDB collection
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection
    
    Returns:
        ChromaDB collection
    """
    try:
        # Try to delete existing collection first
        try:
            client.delete_collection(name=collection_name)
        except:
            pass
        
        # Create a new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "GitHub repository code chunks for RAG"}
        )
        st.success(f"Created new ChromaDB collection: {collection_name}")
        return collection
    
    except Exception as e:
        st.error(f"Error managing collection {collection_name}: {str(e)}")
        
        # Attempt alternative approach
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            st.info(f"Using existing collection: {collection_name}")
            return collection
        except Exception as inner_e:
            st.error(f"Failed to get or create collection: {inner_e}")
            raise

# Modify the main initialization
try:
    # Use the new robust collection management
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client, "github_collection")
except Exception as e:
    st.error(f"Critical error initializing ChromaDB: {str(e)}")
    # Fallback mechanism
    collection = None
chroma_client = get_chroma_client()

# Create or get collection
try:
    collection = chroma_client.get_collection("github_collection")
    st.success("Connected to existing ChromaDB collection.")
except:
    collection = chroma_client.create_collection(
        "github_collection",
        metadata={"description": "GitHub repository code chunks for RAG"}
    )
    st.info("Created new ChromaDB collection.")

# Robust dataset loading function
def extract_text_from_github(num_samples=100, programming_languages=None):
    """
    Extract text from GitHub dataset with comprehensive error handling
    
    Args:
        num_samples (int): Number of samples to extract
        programming_languages (list): Optional list of programming languages to filter
    
    Returns:
        list: List of extracted text samples
    """
    try:
        # First, check network connectivity and dataset availability
        try:
            response = requests.head("https://huggingface.co/datasets/codeparrot/github-code")
            if response.status_code != 200:
                st.error(f"Hugging Face dataset endpoint returned status code {response.status_code}")
                return []
        except requests.RequestException as network_error:
            st.error(f"Network connectivity issue: {network_error}")
            return []

        # Prepare dataset loading parameters
        # Modify language configurations to match the exact config names
        language_configs = {
            'Python': 'Python-all',
            'JavaScript': 'JavaScript-all',
            'Java': 'Java-all',
            'C++': 'C++-all',
            'TypeScript': 'TypeScript-all',
            'Ruby': 'Ruby-all',
            'Go': 'Go-all',
            'Rust': 'Rust-all'
        }

        # If no languages specified, use a default or all available
        if not programming_languages:
            dataset = load_dataset(
                "codeparrot/github-code", 
                split=f"train[:{num_samples}]",
                streaming=False,
                trust_remote_code=True
            )
        else:
            # Map user-friendly language names to correct config names
            valid_configs = [
                language_configs.get(lang, lang + '-all') 
                for lang in programming_languages 
                if lang in language_configs or lang + '-all' in language_configs
            ]

            if not valid_configs:
                st.error("No valid language configurations found.")
                return []

            # Load dataset with specified language configuration
            dataset = load_dataset(
                "codeparrot/github-code", 
                name=valid_configs[0] if len(valid_configs) == 1 else None,
                split=f"train[:{num_samples}]",
                streaming=False,
                trust_remote_code=True
            )
        
        samples = []
        for i, item in enumerate(dataset):
            # More robust content extraction
            content = item.get('code', '')
            if content and isinstance(content, str) and len(content.strip()) > 0:
                samples.append({
                    "id": str(i),
                    "text": content.strip()[:5000],  # Limit text length
                    "metadata": {
                        "language": item.get('language', 'unknown'),
                        "repo_name": item.get('repo_name', 'unknown'),
                        "path": item.get('path', 'unknown')
                    }
                })
        
        return samples
    
    except Exception as e:
        # Comprehensive error logging
        st.error(f"Detailed Error Loading Dataset: {str(e)}")
        st.error(traceback.format_exc())
        
        # Provide more context and potential solutions
        st.info("""
        Comprehensive Troubleshooting for Dataset Loading:
        1. Verify network connectivity
        2. Check Hugging Face dataset availability
        3. Ensure you have the latest versions of:
           - huggingface_hub
           - datasets
           - transformers
        4. Potential workarounds:
           - Try alternative dataset loading method
           - Verify Hugging Face token
        """)
        
        return []
# Manual text chunking function (remains the same as previous implementation)
def chunk_text(text, chunk_size=1000, overlap=200):
    """Chunk text into smaller pieces with overlap"""
    chunks = []
    if len(text) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # If we're not at the beginning, and we can find a space to break at
        if start > 0 and end < len(text):
            # Try to find a space to break at
            space_pos = text.rfind(' ', start, end)
            if space_pos != -1:
                end = space_pos + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    
    return chunks

# Function to generate embeddings (remains the same)
def generate_embeddings(texts):
    """Generate embeddings for a list of texts"""
    return embedding_model.encode(texts).tolist()

# Store documents in ChromaDB (remains largely the same)
def store_in_chromadb(documents):
    """Store chunked documents in ChromaDB"""
    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]
        metadata = doc["metadata"]
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # For each chunk, create an ID and store in ChromaDB
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-chunk-{i}"
            chunk_embedding = generate_embeddings([chunk])[0]
            
            # Add chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            # Add to collection
            collection.add(
                ids=[chunk_id],
                embeddings=[chunk_embedding],
                metadatas=[chunk_metadata],
                documents=[chunk]
            )

# Semantic search function (remains the same)
def semantic_search(query, n_results=3):
    """Perform semantic search on the database"""
    query_embedding = generate_embeddings([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

# RAG generation function (remains largely the same)
def rag_generate(query, n_chunks=3, temperature=0.7):
    """
    Perform RAG by retrieving relevant chunks and generating response
    using OpenAI API directly
    """
    # Get relevant chunks
    search_results = semantic_search(query, n_results=n_chunks)
    
    retrieved_chunks = search_results["documents"][0]
    
    # Create context from chunks
    context = "\n\n".join([f"CHUNK {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    # Construct prompt with retrieved context
    prompt = f"""
I want you to answer the following query based on the provided context.
If the context doesn't contain relevant information, just say so.

CONTEXT:
{context}

QUERY:
{query}

ANSWER:
"""
    
    try:
        # Use OpenAI API directly
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=300
        )
        
        generated_response = response.choices[0].message.content
        
        return {
            "response": generated_response,
            "retrieved_chunks": retrieved_chunks
        }
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return {
            "response": "Sorry, I couldn't generate a response.",
            "retrieved_chunks": retrieved_chunks
        }

# Streamlit UI
def main():
    st.title("RAG Application with GitHub Code Dataset")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Load Data", "Search", "Settings"])

    with tab1:
        st.header("Load Data")
        st.write("Load and chunk data from GitHub repositories")
        
        # Language selection multiselect
        languages = st.multiselect(
            "Select Programming Languages", 
            ['Python', 'JavaScript', 'Java', 'C++', 'TypeScript', 'Ruby', 'Go', 'Rust']
        )
        
        num_samples = st.slider("Number of samples to load", 10, 500, 50)
        
        if st.button("Load GitHub Dataset"):
            with st.spinner("Loading and processing dataset..."):
                documents = extract_text_from_github(
                    num_samples, 
                    programming_languages=languages if languages else None
                )
                st.session_state.documents = documents
                st.success(f"Loaded {len(documents)} documents from GitHub dataset")
        
        if st.button("Process and Store in ChromaDB"):
            if hasattr(st.session_state, 'documents'):
                with st.spinner("Processing and storing documents..."):
                    store_in_chromadb(st.session_state.documents)
                    st.success(f"Successfully processed and stored documents in ChromaDB")
            else:
                st.error("Please load the dataset first")
        
        # Show collection info
        try:
            count = collection.count()
            st.info(f"ChromaDB collection currently has {count} chunks")
        except:
            st.warning("ChromaDB collection not yet initialized")

    # Search and Settings tabs remain the same as in the previous implementation
    with tab2:
        st.header("Search and Generate")
        
        query = st.text_area("Enter your query:", height=100)
        n_chunks = st.slider("Number of chunks to retrieve", 1, 10, 3)
        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.7)
        
        if st.button("Generate Answer"):
            if query:
                with st.spinner("Generating answer..."):
                    try:
                        result = rag_generate(query, n_chunks, temperature)
                        
                        st.subheader("Generated Answer")
                        st.write(result["response"])
                        
                        with st.expander("Show retrieved chunks"):
                            for i, chunk in enumerate(result["retrieved_chunks"]):
                                st.markdown(f"**Chunk {i+1}**")
                                st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                                st.divider()
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.error("Please enter a query")

    with tab3:
        st.header("Settings")
        
        st.subheader("Chunking Parameters")
        chunk_size = st.slider("Chunk size (characters)", 500, 5000, 1000)
        overlap = st.slider("Overlap size (characters)", 0, 500, 200)
        
        st.subheader("API Keys")
        api_key = st.text_input("OpenAI API Key", type="password")
        if st.button("Save API Key"):
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API Key saved for this session")
        
        st.subheader("Reset Database")
        if st.button("Clear ChromaDB Collection", type="primary"):
            try:
                chroma_client.delete_collection("github_collection")
                collection = chroma_client.create_collection(
                    "github_collection",
                    metadata={"description": "GitHub repository code chunks for RAG"}
                )
                st.success("ChromaDB collection cleared and recreated")
            except Exception as e:
                st.error(f"Error clearing collection: {str(e)}")

# Show app instructions in sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. **Select Languages**: Choose programming languages (optional)
2. **Load Data**: Load data from GitHub dataset
3. **Process and Store**: Click to process and store in ChromaDB
4. **Search**: Enter a query and generate responses using RAG

This app:
- Extracts code using the datasets library
- Chunks code without using a chunking library
- Stores chunks in ChromaDB
- Performs semantic search for retrieval 
- Uses retrieved context for LLM generation
""")

# Run the app
if __name__ == "__main__":
    main()
