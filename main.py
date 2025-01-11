import os
#os.environ["OPENAI_API_KEY"] = "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"
# Run with streamlit : streamlit run main.py
from typing import List, Dict, Any
import PyPDF2
import docx
import pptx
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import json
from datetime import datetime


class DocumentProcessor:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.metadata_store = {}

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from document files"""
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        created_time = os.path.getctime(filepath)

        metadata = {
            "filename": filename,
            "filepath": filepath,
            "file_type": file_ext,
            "created_date": datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S'),
            "size_kb": os.path.getsize(filepath) / 1024
        }

        return metadata

    def process_document(self, filepath: str) -> List[str]:
        """Process different document types and extract text"""
        file_ext = os.path.splitext(filepath)[1].lower()

        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(filepath)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(filepath)
            elif file_ext in ['.pptx', '.ppt']:
                loader = UnstructuredPowerPointLoader(filepath)
            else:
                return []

            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)

            # Store metadata
            metadata = self.extract_metadata(filepath)
            self.metadata_store[filepath] = metadata

            return texts
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return []

    def scan_directory(self) -> List[str]:
        """Recursively scan directory and process all documents"""
        all_texts = []

        for root, _, files in os.walk(self.base_folder):
            for file in files:
                filepath = os.path.join(root, file)
                if file.endswith(('.pdf', '.docx', '.pptx', '.ppt')):
                    texts = self.process_document(filepath)
                    all_texts.extend(texts)

        return all_texts


class AIQueryEngine:
    def __init__(self, texts: List[str]):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(texts, self.embeddings)
        self.chat_model = ChatOpenAI(temperature=0.7)
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.chat_model,
            self.vectorstore.as_retriever(),
            return_source_documents=True
        )
        self.chat_history = []

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return relevant information"""
        result = self.qa_chain({"question": question, "chat_history": self.chat_history})

        # Update chat history
        self.chat_history.append((question, result["answer"]))

        # Get source documents
        sources = [doc.metadata for doc in result["source_documents"]]

        return {
            "answer": result["answer"],
            "sources": sources
        }


def create_streamlit_app():
    st.title("AI-Powered Document Management System")

    # Initialize session state
    if 'qa_engine' not in st.session_state:
        st.session_state.qa_engine = None

    # Folder selection
    folder_path = st.text_input("Enter folder path containing documents:")

    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            processor = DocumentProcessor(folder_path)
            texts = processor.scan_directory()
            st.session_state.qa_engine = AIQueryEngine(texts)
            st.success("Documents processed successfully!")

            # Save metadata
            with open('metadata_store.json', 'w') as f:
                json.dump(processor.metadata_store, f, indent=4, default=str)

    # Query interface
    if st.session_state.qa_engine:
        question = st.text_input("Ask a question about your documents:")

        if st.button("Submit"):
            with st.spinner("Processing query..."):
                result = st.session_state.qa_engine.query(question)

                st.write("Answer:", result["answer"])

                st.write("Sources:")
                for source in result["sources"]:
                    st.write(f"- {source.get('filename', 'Unknown file')}")


if __name__ == "__main__":
    create_streamlit_app()