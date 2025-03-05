import streamlit as st
import os
import sys
import traceback
import requests

# SQLite Patch
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
import torch
import openai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ChromaDB Client Initialization
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(
        path=".chromadb",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

# Robust collection management function
def get_or_create_collection(client, collection_name):
    """
    Safely get or create a ChromaDB collection
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

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize ChromaDB
try:
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client, "github_collection")
except Exception as e:
    st.error(f"Critical error initializing ChromaDB: {str(e)}")
    collection = None

# Dataset loading function for GitHub Code
def extract_text_from_github(num_samples=100, programming_languages=None):
    """
    Enhanced GitHub dataset loading with comprehensive error handling and logging
    """
    try:
        # Detailed network and library version checks
        import sys
        import pkg_resources

        st.write("Python Version:", sys.version)
        st.write("Installed Library Versions:")
        for package in ['huggingface_hub', 'datasets', 'transformers']:
            try:
                version = pkg_resources.get_distribution(package).version
                st.write(f"{package}: {version}")
            except pkg_resources.DistributionNotFound:
                st.error(f"{package} is not installed")

        # Network connectivity test
        try:
            response = requests.get("https://huggingface.co", timeout=10)
            st.success(f"Network connectivity to Hugging Face: {response.status_code}")
        except requests.RequestException as network_error:
            st.error(f"Network connectivity issue: {network_error}")
            return []

        # Comprehensive dataset loading with verbose logging
        try:
            # Use an authentication token if available
            from huggingface_hub import login
            if hasattr(st.session_state, 'hf_token'):
                login(token=st.session_state.hf_token)
                st.info("Logged in with Hugging Face token")

            # Language configuration logic remains the same as in original code
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

            # Configuration selection logic
            if not programming_languages:
                st.info("Loading default dataset configuration")
                dataset = load_dataset(
                    "codeparrot/github-code", 
                    split=f"train[:{num_samples}]",
                    streaming=False,
                    trust_remote_code=True
                )
            else:
                valid_configs = [
                    language_configs.get(lang, lang + '-all') 
                    for lang in programming_languages 
                    if lang in language_configs or lang + '-all' in language_configs
                ]

                if not valid_configs:
                    st.error("No valid language configurations found.")
                    return []

                st.info(f"Using configurations: {valid_configs}")
                dataset = load_dataset(
                    "codeparrot/github-code", 
                    name=valid_configs[0] if len(valid_configs) == 1 else None,
                    split=f"train[:{num_samples}]",
                    streaming=False,
                    trust_remote_code=True
                )
            
            # Rest of the processing remains the same
            samples = []
            for i, item in enumerate(dataset):
                content = item.get('code', '')
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    samples.append({
                        "id": str(i),
                        "text": content.strip()[:5000],
                        "metadata": {
                            "language": item.get('language', 'unknown'),
                            "repo_name": item.get('repo_name', 'unknown'),
                            "path": item.get('path', 'unknown')
                        }
                    })
            
            st.success(f"Successfully loaded {len(samples)} samples")
            return samples

        except Exception as dataset_error:
            st.error(f"Dataset loading error: {str(dataset_error)}")
            st.error(traceback.format_exc())
            return []

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.error(traceback.format_exc())
        return []

# Manual text chunking function
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

# Function to generate embeddings
def generate_embeddings(texts):
    """Generate embeddings for a list of texts"""
    return embedding_model.encode(texts).tolist()

# Store documents in ChromaDB
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

# Semantic search function
def semantic_search(query, n_results=3):
    """Perform semantic search on the database"""
    query_embedding = generate_embeddings([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

# RAG generation function
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

# Optional: Debug function to print available configurations
def print_github_dataset_configs():
    from datasets import get_dataset_config_names
    
    configs = get_dataset_config_names("codeparrot/github-code")
    st.write("Available GitHub Code Dataset Configurations:")
    for config in configs:
        st.write(config)

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
            ['Python', 'JavaScript', 'Java', 'C++', 
             'TypeScript', 'Ruby', 'Go', 'Rust']
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
                collection = get_or_create_collection(chroma_client, "github_collection")
                st.success("ChromaDB collection cleared and recreated")
            except Exception as e:
                st.error(f"Error clearing collection: {str(e)}")

        st.subheader("Dataset Configurations")
        if st.button("Show Available Configurations"):
            print_github_dataset_configs()

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
