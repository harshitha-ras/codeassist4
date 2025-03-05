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

# ChromaDB and other previous utility functions remain the same
# [Keep all previous functions like get_chroma_client(), parse_text_file(), etc.]

# Modify the OpenAI client initialization
def create_openai_client(api_key):
    """
    Create OpenAI client with the provided API key
    """
    if not api_key:
        raise ValueError("API key cannot be empty")
    
    try:
        client = OpenAI(api_key=api_key)
        # Verify the API key by making a simple API call
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return client
    except Exception as e:
        raise ValueError(f"Invalid API key: {str(e)}")

# Streamlit UI with improved API key handling
def main():
    st.title("Document RAG Application")
    
    # Global variable to store the OpenAI client
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None
    
    # API Key Input Section
    st.sidebar.header("OpenAI API Key")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key", 
        type="password", 
        key="openai_api_key_input"
    )
    
    # Validate and set API key
    if api_key:
        try:
            st.session_state.openai_client = create_openai_client(api_key)
            st.sidebar.success("API Key validated successfully!")
        except ValueError as e:
            st.sidebar.error(str(e))
            st.session_state.openai_client = None
    
    # Check if client is initialized before proceeding
    if not st.session_state.openai_client:
        st.warning("Please enter a valid OpenAI API Key to use the application.")
        return
    
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

    with tab1:
        st.header("Upload Document")
        
        # File uploader with multiple file type support
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['txt', 'pdf', 'docx', 'py'],
            help="Upload a text, PDF, Word, or Python document"
        )
        
        # Preprocessing option for Python files
        preprocess_python = st.checkbox("Preprocess Python file (remove comments)")
        
        if uploaded_file is not None:
            # Determine file type and parse accordingly
            try:
                if uploaded_file.type == 'text/plain':
                    document_text = parse_text_file(uploaded_file)
                elif uploaded_file.type == 'application/pdf':
                    document_text = parse_pdf_file(uploaded_file)
                elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    document_text = parse_docx_file(uploaded_file)
                elif uploaded_file.type == 'text/x-python':
                    document_text = parse_python_file(uploaded_file)
                    
                    # Optional preprocessing for Python files
                    if preprocess_python:
                        document_text = preprocess_python_code(document_text)
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
                        # Use the session state client for generation
                        result = rag_generate(query, n_chunks, temperature, st.session_state.openai_client)
                        
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

# Modify RAG generation to accept the client as a parameter
def rag_generate(query, n_chunks=3, temperature=0.7, client=None):
    """
    Perform RAG by retrieving relevant chunks and generating response
    using OpenAI API directly
    """
    # Ensure we have a valid client
    if not client:
        st.error("OpenAI client not initialized. Please provide an API key.")
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
        response = client.chat.completions.create(
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
