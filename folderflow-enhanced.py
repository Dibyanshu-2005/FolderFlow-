from pathlib import Path
import PyPDF2
import docx
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import faiss
import openai
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import json

class EnhancedAI:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an AI response using context and query."""
        # Prepare context from relevant documents
        context_text = "\n\n".join([f"From {doc['source']}:\n{doc['text']}" for doc in context])
        
        # Create system message
        system_message = """You are a technical assistant that helps users find information in their documents. 
        Use the provided context to answer questions accurately. Always cite your sources.
        If you're not sure about something, say so."""
        
        # Create the conversation
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        try:
            # Get response from GPT
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract sources from context
            sources = list(set(doc['source'] for doc in context))
            
            return {
                "answer": response.choices[0].message.content,
                "sources": sources,
                "confidence": response.choices[0].finish_reason == "stop"
            }
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

class DocumentProcessor:
    def __init__(self, openai_api_key: str):
        self.ai = EnhancedAI(openai_api_key)
        self.index = faiss.IndexFlatL2(384)
        self.documents = []
        
    def process_chunk(self, text: str, source: str) -> None:
        """Process a chunk of text."""
        embedding = self.ai.embedder.encode([text])[0]
        self.index.add(np.array([embedding]).astype('float32'))
        self.documents.append({
            'text': text,
            'source': source,
            'embedding_id': len(self.documents)
        })

    def process_document(self, file_path: str) -> None:
        """Process a single document."""
        text = ""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            
            # Split into chunks (considering token limits)
            chunks = self.split_into_chunks(text)
            
            # Process chunks in parallel
            with ThreadPoolExecutor() as executor:
                executor.map(
                    lambda chunk: self.process_chunk(chunk, str(file_path)),
                    chunks
                )
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def split_into_chunks(self, text: str, max_tokens: int = 1000) -> List[str]:
        """Split text into chunks while respecting sentence boundaries."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.ai.tokenizer.encode(sentence))
            if current_length + sentence_tokens > max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        query_embedding = self.ai.embedder.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        return [self.documents[idx] for idx in I[0] if idx < len(self.documents)]

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Answer questions using AI and retrieved documents."""
        relevant_docs = self.search(query)
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information in the documents.",
                "sources": [],
                "confidence": 0.0
            }
        
        return self.ai.generate_response(query, relevant_docs)

class FolderFlow:
    def __init__(self, openai_api_key: str):
        self.processor = DocumentProcessor(openai_api_key)
        
    def initialize(self, folder_path: str):
        """Initialize by processing all documents in a folder."""
        for file_path in Path(folder_path).rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.docx']:
                print(f"Processing {file_path}")
                self.processor.process_document(str(file_path))
        print("Initialization complete!")
        
    def ask(self, query: str) -> Dict[str, Any]:
        """Process a user query."""
        try:
            result = self.processor.answer_question(query)
            return {
                "status": "success",
                "response": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)
    
    # Initialize FolderFlow
    folderflow = FolderFlow(api_key)
    
    # Process documents
    folderflow.initialize("./documents")
    
    # Interactive chat loop
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        response = folderflow.ask(query)
        if response["status"] == "success":
            print("\nAnswer:", response["response"]["answer"])
            print("\nSources:", ", ".join(response["response"]["sources"]))
        else:
            print("\nError:", response["error"])
