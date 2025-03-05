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

# Document parsing functions
def parse_text_file(uploaded_file):
    """Parse plain text files"""
    return uploaded_file.getvalue().decode('utf-8')

def parse_python_file(uploaded_file):
    """Parse Python files, preserving code structure and comments"""
    try:
        # Read the Python file content
        python_content = uploaded_file.getvalue().decode('utf-8')
        
        # Optional: Add some basic preprocessing
        # Remove excessive whitespace while preserving meaningful formatting
        cleaned_content = '\n'.join(line.rstrip() for line in python_content.split('\n'))
        
        return cleaned_content
    except Exception as e:
        st.error(f"Error parsing Python file: {e}")
        return ""

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

# The rest of the previous script remains the same, with one modification in the main() function
# Update the file uploader type and parsing logic

# In the main() function, modify the file upload section:
def main():
    st.title("Document RAG Application")
    
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

    # Rest of the function remains the same
    # ... (previous implementation)

# Optionally, add some Python-specific preprocessing
def preprocess_python_code(code):
    """
    More advanced Python code preprocessing
    - Remove comments
    - Normalize whitespace
    - Optionally remove docstrings
    """
    import re
    
    # Remove single-line comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Remove multi-line comments (docstrings)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    # Normalize whitespace
    code = '\n'.join(line.rstrip() for line in code.split('\n') if line.strip())
    
    return code

# The rest of the script remains the same as in the previous implementation
