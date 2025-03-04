__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import chromadb
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAI

# Set your API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# App title and description
st.title("RAG Application with Stack Database")
st.write("This app performs retrieval-augmented generation using the Stack database.")

# Initialize sentence transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize ChromaDB
@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

chroma_client = get_chroma_client()

# Create or get collection
try:
    collection = chroma_client.get_collection("stack_collection")
    st.success("Connected to existing ChromaDB collection.")
except:
    collection = chroma_client.create_collection(
        "stack_collection",
        metadata={"description": "Stack dataset chunks for RAG"}
    )
    st.info("Created new ChromaDB collection.")

# Function to extract text from Stack dataset
def extract_text_from_stack(num_samples=100):
    """Load and extract text from Stack dataset using Hugging Face datasets library"""
    dataset = load_dataset("bigcode/the-stack", streaming=True, split="train")
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        content = item.get('content', '')
        if content and isinstance(content, str):
            samples.append({
                "id": str(i),
                "text": content,
                "metadata": {
                    "lang": item.get('lang', ''),
                    "repo": item.get('repo_name', '')
                }
            })
    
    return samples

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

# RAG function - retrieve context and generate response
def rag_generate(query, n_chunks=3, temperature=0.7):
    """Perform RAG by retrieving relevant chunks and generating response"""
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
    
    # Use OpenAI for generation
    llm = OpenAI(temperature=temperature)
    response = llm.invoke(prompt)
    
    return {
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }

# UI Components
tab1, tab2, tab3 = st.tabs(["Load Data", "Search", "Settings"])

with tab1:
    st.header("Load Data")
    st.write("Load and chunk data from the Stack dataset")
    
    num_samples = st.slider("Number of samples to load", 10, 500, 50)
    
    if st.button("Load Stack Dataset"):
        with st.spinner("Loading and processing dataset..."):
            documents = extract_text_from_stack(num_samples)
            st.session_state.documents = documents
            st.success(f"Loaded {len(documents)} documents from Stack dataset")
    
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
            chroma_client.delete_collection("stack_collection")
            collection = chroma_client.create_collection(
                "stack_collection",
                metadata={"description": "Stack dataset chunks for RAG"}
            )
            st.success("ChromaDB collection cleared and recreated")
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")

# Show app instructions in sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. **Load Data**: First load data from the Stack dataset
2. **Process and Store**: Click to process and store in ChromaDB
3. **Search**: Enter a query and generate responses using RAG

This app:
- Extracts text using the datasets library
- Manually chunks text without using a chunking library
- Stores chunks in ChromaDB
- Performs semantic search for retrieval 
- Uses retrieved context for LLM generation
""")
