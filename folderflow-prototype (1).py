# Core implementation of Folderflow
from pathlib import Path
import PyPDF2
import docx
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import faiss
import os

class DocumentProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        self.index = faiss.IndexFlatL2(384)  # Dimension matches the embedder
        self.documents = []
        
    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def process_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text

    def process_folder(self, folder_path: str):
        """Recursively process all documents in a folder."""
        for file_path in Path(folder_path).rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.docx']:
                try:
                    text = ""
                    if file_path.suffix.lower() == '.pdf':
                        text = self.process_pdf(str(file_path))
                    else:
                        text = self.process_docx(str(file_path))
                    
                    # Split text into chunks for better processing
                    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                    
                    for chunk in chunks:
                        # Create embedding
                        embedding = self.embedder.encode([chunk])[0]
                        
                        # Add to FAISS index
                        self.index.add(np.array([embedding]).astype('float32'))
                        
                        # Store document info
                        self.documents.append({
                            'text': chunk,
                            'source': str(file_path),
                            'embedding_id': len(self.documents)
                        })
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic search."""
        query_embedding = self.embedder.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        results = []
        for idx in I[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Answer questions using retrieved documents."""
        relevant_docs = self.search(query)
        if not relevant_docs:
            return {"answer": "I couldn't find relevant information to answer your question.",
                    "source": None}

        # Combine relevant documents
        context = " ".join([doc['text'] for doc in relevant_docs])
        
        # Get answer using QA pipeline
        qa_result = self.qa_pipeline(question=query, context=context)
        
        return {
            "answer": qa_result['answer'],
            "confidence": qa_result['score'],
            "source": relevant_docs[0]['source']  # Source of the most relevant document
        }

class ChatInterface:
    def __init__(self):
        self.processor = DocumentProcessor()
        
    def initialize(self, folder_path: str):
        """Initialize by processing documents in the given folder."""
        self.processor.process_folder(folder_path)
        print("Initialization complete. Ready to answer questions!")
        
    def chat(self, query: str) -> str:
        """Handle user queries and return responses."""
        try:
            result = self.processor.answer_question(query)
            response = f"Answer: {result['answer']}\n"
            response += f"Source: {result['source']}\n"
            response += f"Confidence: {result['confidence']:.2f}"
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

# Example usage
if __name__ == "__main__":
    chat_interface = ChatInterface()
    
    # Initialize with your document folder
    chat_interface.initialize("path/to/your/documents")
    
    # Interactive chat loop
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = chat_interface.chat(query)
        print("\n" + response + "\n")
