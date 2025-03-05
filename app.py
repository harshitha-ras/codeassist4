import streamlit as st
import os
import sys
import traceback
import requests
import io
import PyPDF2  # PDF parsing
import docx  # Word document parsing
import re  # For code preprocessing

# SQLite Patch
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
import torch
from openai import OpenAI  # Updated import
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

# [... rest of the previous parsing and utility functions remain the same ...]

# OpenAI Client Initialization with careful error handling
def initialize_openai_client():
    """
    Carefully initialize OpenAI client with multiple fallback methods
    """
    try:
        # First, try Streamlit secrets
        if hasattr(st.secrets, 'openai') and 'api_key' in st.secrets.openai:
            return OpenAI(api_key=st.secrets.openai.api_key)
        
        # Then try environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        
        # If no key found, return None
        return None
    
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

# Initialize the OpenAI client
openai_client = initialize_openai_client()

# API Key validation and input UI
def validate_and_set_api_key():
    global openai_client
    
    # Check if client is not initialized
    if not openai_client:
        st.sidebar.warning("OpenAI API Key is required!")
        api_key_input = st.sidebar.text_input(
            "Enter your OpenAI API Key", 
            type="password"
        )
        
        if api_key_input:
            try:
                # Attempt to create client with input key
                openai_client = OpenAI(api_key=api_key_input)
                st.sidebar.success("API Key set successfully!")
                return True
            except Exception as e:
                st.sidebar.error(f"Invalid API Key: {e}")
                return False
    
    return True

# [... rest of the previous code remains the same, just replace the OpenAI client initialization part ...]

# Streamlit UI
def main():
    st.title("Document RAG Application")
    
    # Validate API key first
    if not validate_and_set_api_key():
        return
    
    # Rest of the main() function remains the same...
    # Sidebar instructions
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a document (TXT, PDF, DOCX, PY)
    2. Enter a query about the document
    3. Get AI-generated answers based on document content
    
    Tips:
    - Ensure a clear, specific query
    - Works best with well-structured documents
    - OpenAI API key required
    """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["Upload Document", "Search and Generate"])

    # [... rest of the main function remains the same ...]

# RAG generation function
def rag_generate(query, n_chunks=3, temperature=0.7):
    """
    Perform RAG by retrieving relevant chunks and generating response
    using OpenAI API directly
    """
    # Ensure we have a valid client
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return {
            "response": "Sorry, OpenAI client initialization failed.",
            "retrieved_chunks": []
        }
    
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
        # Use OpenAI API with new method
        response = openai_client.chat.completions.create(
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

# Run the app
if __name__ == "__main__":
    main()
