import streamlit as st
import os
import sys
import traceback
import requests
import io
import PyPDF2  # PDF parsing
import docx  # Word document parsing

# SQLite Patch
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
import torch
import openai
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
            metadata={"description": "User uploaded document chunks for RAG"}
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

# Document parsing functions
def parse_text_file(uploaded_file):
    """Parse plain text files"""
    return uploaded_file.getvalue().decode('utf-8')

def parse_pdf_file(uploaded_file):
    """Parse PDF files"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return ""

def parse_docx_file(uploaded_file):
    """Parse Word documents"""
    try:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        st.error(f"Error parsing DOCX: {e}")
        return ""

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize ChromaDB
try:
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client, "document_collection")
except Exception as e:
    st.error(f"Critical error initializing ChromaDB: {str(e)}")
    collection = None

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
    # Clear existing collection
    try:
        collection.delete(where={})
    except:
        pass
    
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

# OpenAI API setup
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate OpenAI API key
if not openai.api_key:
    st.sidebar.warning("OpenAI API Key is required!")
    api_key_input = st.sidebar.text_input(
        "Enter your OpenAI API Key", 
        type="password"
    )
    
    if api_key_input:
        openai.api_key = api_key_input
        st.sidebar.success("API Key set successfully!")

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

# Streamlit UI
def main():
    st.title("Document RAG Application")
    
    # Sidebar instructions
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a document (TXT, PDF, DOCX)
    2. Enter a query about the document
    3. Get AI-generated answers based on document content
    
    Tips:
    - Ensure a clear, specific query
    - Works best with well-structured documents
    - OpenAI API key required
    """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["Upload Document", "Search and Generate"])

    with tab1:
        st.header("Upload Document")
        
        # File uploader with multiple file type support
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['txt', 'pdf', 'docx'],
            help="Upload a text, PDF, or Word document"
        )
        
        if uploaded_file is not None:
            # Determine file type and parse accordingly
            try:
                if uploaded_file.type == 'text/plain':
                    document_text = parse_text_file(uploaded_file)
                elif uploaded_file.type == 'application/pdf':
                    document_text = parse_pdf_file(uploaded_file)
                elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    document_text = parse_docx_file(uploaded_file)
                else:
                    st.error("Unsupported file type")
                    return
                
                # Prepare document for ChromaDB
                documents = [{
                    "id": "uploaded_doc",
                    "text": document_text,
                    "metadata": {
                        "filename": uploaded_file.name,
                        "file_type": uploaded_file.type
                    }
                }]
                
                # Process and store document
                with st.spinner("Processing document..."):
                    store_in_chromadb(documents)
                    st.success(f"Successfully processed {uploaded_file.name}")
                    
                    # Show document preview
                    with st.expander("Document Preview"):
                        st.text(document_text[:1000] + "..." if len(document_text) > 1000 else document_text)
            
            except Exception as e:
                st.error(f"Error processing document: {e}")

    with tab2:
        st.header("Search and Generate")
        
        # Validate that a document has been uploaded
        if collection.count() == 0:
            st.warning("Please upload a document first")
            return
        
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

# Run the app
if __name__ == "__main__":
    main()
