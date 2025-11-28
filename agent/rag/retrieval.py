from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class DocumentChunk:
    def __init__(self, content: str, source: str, chunk_id: str):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id

    def __repr__(self):
        return f'<Chunk "{self.chunk_id}": "{self.content[:50]}"...>'

class SimpleRetriever:
    def __init__(self, docs_path: str = "docs/"):
       self.docs_path = Path(docs_path)
       self.chunks: List[DocumentChunk] = []
       self.vectorizer = TfidfVectorizer(stop_words="english")
       self.tfidf_matrix = None

       self._load_documents()

    def _load_documents(self):
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs folder not found at {self.docs_path}")
        
        for doc_file in self.docs_path.glob("*.md"):
            self._chunk_document(doc_file)
        
        # Build TF-IDF matrix
        if self.chunks:
            texts = [chunk.content for chunk in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print(f"Loaded {len(self.chunks)} chunks from {len(list(self.docs_path.glob('*.md')))} documents")
    
    def _chunk_document(self, file_path: Path):
        content = file_path.read_text(encoding="utf-8")
        source_name = file_path.stem
        raw_chunks = re.split(r'\n\s*\n+', content)
        
        chunk_counter = 0
        for raw_chunk in raw_chunks:
            raw_chunk = raw_chunk.strip()
            if len(raw_chunk) < 10:
                continue
            chunk_id = f'{source_name}::chunk{chunk_counter}'
            self.chunks.append(DocumentChunk(content=raw_chunk, source=source_name, chunk_id=chunk_id))
            chunk_counter += 1
            
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "content": chunk.content,
                "score": float(similarities[idx])
            })
        return results

    def get_all_chunks_ids(self):
        return [chunk.chunk_id for chunk in self.chunks]


# Test code
if __name__ == "__main__":
    print("Testing SimpleRetriever...\n")
    
    retriever = SimpleRetriever(docs_path="docs")
    
    print("=== TEST 1: Loaded Chunks ===")
    print(f"Total chunks: {len(retriever.chunks)}")
    print("Sample chunks:")
    for chunk in retriever.chunks[:3]:
        print(f"  - {chunk.chunk_id}: {chunk.content[:60]}...")
    print()
    
    print("=== TEST 2: Search 'return policy beverages' ===")
    results = retriever.retrieve("return policy beverages", top_k=2)
    for result in results:
        print(f"Chunk: {result['chunk_id']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:100]}...")
        print()
    
    print("=== TEST 3: Search 'summer 1997 marketing' ===")
    results = retriever.retrieve("summer 1997 marketing", top_k=2)
    for result in results:
        print(f"Chunk: {result['chunk_id']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:100]}...")
        print()
    
    print("=== TEST 4: Search 'average order value AOV' ===")
    results = retriever.retrieve("average order value AOV", top_k=2)
    for result in results:
        print(f"Chunk: {result['chunk_id']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:100]}...")
        print()