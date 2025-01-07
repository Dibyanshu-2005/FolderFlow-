import os
from pathlib import Path
import json
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import re

class Document:
    def __init__(self, path: str, content: str, metadata: Dict = None):
        self.path = path
        self.content = content
        self.metadata = metadata or {}
        self.embedding = None
        self.created_date = datetime.now()
        self.last_modified = datetime.now()
        
    def extract_technical_params(self) -> Dict:
        """Extract technical parameters using regex patterns."""
        patterns = {
            'dimensions': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m)',
            'temperature': r'(\d+(?:\.\d+)?)\s*(?:°C|°F)',
            'pressure': r'(\d+(?:\.\d+)?)\s*(?:Pa|kPa|MPa)',
            'voltage': r'(\d+(?:\.\d+)?)\s*(?:V|kV)',
        }
        
        params = {}
        for param_type, pattern in patterns.items():
            matches = re.findall(pattern, self.content)
            if matches:
                params[param_type] = matches
        return params

class FolderFlow:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.documents: Dict[str, Document] = {}
        self.index = {}
        self.relationships = {}
        
    def index_folder(self, folder_path: Optional[str] = None) -> None:
        """Index all documents in the specified folder and subfolders."""
        if folder_path is None:
            folder_path = self.base_path
        else:
            folder_path = Path(folder_path)
            
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.pdf']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = Document(
                        path=str(file_path.relative_to(self.base_path)),
                        content=content
                    )
                    
                    # Extract technical parameters
                    doc.metadata['technical_params'] = doc.extract_technical_params()
                    
                    # Create simple embedding (in practice, use proper embedding model)
                    doc.embedding = self._create_simple_embedding(content)
                    
                    self.documents[doc.path] = doc
                    self._update_index(doc)
                    self._find_relationships(doc)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    def _create_simple_embedding(self, content: str) -> np.ndarray:
        """Create a simple document embedding (placeholder for actual embedding model)."""
        # In practice, use a proper embedding model
        words = set(content.lower().split())
        return np.random.rand(128)  # Simulate 128-dimensional embedding
    
    def _update_index(self, document: Document) -> None:
        """Update the search index with document terms."""
        terms = document.content.lower().split()
        for term in terms:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(document.path)
    
    def _find_relationships(self, document: Document) -> None:
        """Find relationships between documents based on content similarity."""
        for other_path, other_doc in self.documents.items():
            if other_path != document.path:
                similarity = self._calculate_similarity(document, other_doc)
                if similarity > 0.7:  # Threshold for relationship
                    if document.path not in self.relationships:
                        self.relationships[document.path] = set()
                    self.relationships[document.path].add(other_path)
    
    def _calculate_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate similarity between two documents using their embeddings."""
        if doc1.embedding is None or doc2.embedding is None:
            return 0.0
        return float(np.dot(doc1.embedding, doc2.embedding))
    
    def search(self, query: str) -> List[Dict]:
        """Search for documents matching the query."""
        query_terms = query.lower().split()
        matching_docs = set()
        
        # Find documents containing query terms
        for term in query_terms:
            if term in self.index:
                if not matching_docs:
                    matching_docs = self.index[term].copy()
                else:
                    matching_docs &= self.index[term]
        
        # Prepare results
        results = []
        for doc_path in matching_docs:
            doc = self.documents[doc_path]
            results.append({
                'path': doc.path,
                'preview': doc.content[:200] + '...',
                'technical_params': doc.metadata.get('technical_params', {}),
                'related_docs': list(self.relationships.get(doc.path, set())),
                'last_modified': doc.last_modified.isoformat()
            })
        
        return sorted(results, key=lambda x: x['last_modified'], reverse=True)

# Example usage
if __name__ == "__main__":
    # Initialize FolderFlow
    folderflow = FolderFlow("./technical_docs")
    
    # Index documents
    folderflow.index_folder()
    
    # Perform a search
    results = folderflow.search("pressure sensor calibration")
    
    # Print results
    print(json.dumps(results, indent=2))
