import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

class ImprovedRAG:
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 llama_url: str = "http://localhost:8080",
                 index_dir: str = "./rag_index",
                 documents_dir: str = "./documents",
                 max_context_tokens: int = 1200):  # NEW: Maximum context tokens
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llama_url = llama_url
        self.index_dir = index_dir
        self.documents_dir = documents_dir
        self.max_context_tokens = max_context_tokens  # NEW: Token limit
        self.index = None
        self.chunks = []
        self.metadata = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Create directories
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(documents_dir, exist_ok=True)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 0.75 words)"""
        return int(len(text.split()) * 1.33)
    
    def smart_chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 30) -> List[str]:  # REDUCED chunk size
        """Improved text chunking that respects sentence boundaries"""
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_words > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_words
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    
    def load_documents_from_directory(self) -> List[Dict[str, str]]:
        """Load all text files from documents directory with better metadata"""
        documents = []
        
        if not os.path.exists(self.documents_dir):
            print(f"Documents directory {self.documents_dir} not found!")
            return documents
        
        # Supported file types
        supported_extensions = ['.txt', '.md', '.text']
        
        for filename in os.listdir(self.documents_dir):
            file_path = os.path.join(self.documents_dir, filename)
            
            # Skip directories and non-text files
            if not os.path.isfile(file_path):
                continue
                
            _, ext = os.path.splitext(filename.lower())
            if ext not in supported_extensions:
                print(f"Skipping unsupported file: {filename}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():  # Only add non-empty files
                    # Extract first line as potential title
                    lines = content.strip().split('\n')
                    first_line = lines[0].strip()
                    
                    # Use first line as title if it's short and looks like a title
                    title = first_line if len(first_line) < 100 and not first_line.endswith('.') else os.path.splitext(filename)[0]
                    
                    documents.append({
                        "content": content,
                        "filename": filename,
                        "title": title,
                        "file_size": len(content),
                        "word_count": len(content.split())
                    })
                    print(f"Loaded: {filename} ({len(content)} characters, {len(content.split())} words)")
                else:
                    print(f"Skipping empty file: {filename}")
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        print(f"Loaded {len(documents)} documents from {self.documents_dir}")
        return documents
    
    def process_documents(self, documents: List[Dict[str, str]]):
        """Process documents with improved chunking and dual indexing"""
        print("Processing documents with improved chunking...")
        all_chunks = []
        all_metadata = []
        
        for doc in documents:
            content = doc["content"]
            filename = doc.get("filename", "unknown")
            title = doc.get("title", filename)
            
            # Use smaller chunks to prevent context overflow
            chunks = self.smart_chunk_text(content, chunk_size=256, overlap=30)  # REDUCED
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "filename": filename,
                    "title": title,
                    "chunk_id": i,
                    "text": chunk,
                    "chunk_length": len(chunk),
                    "word_count": len(chunk.split()),
                    "doc_word_count": doc.get("word_count", 0),
                    "position_in_doc": i / len(chunks)  # Relative position in document
                })
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Create FAISS index with better configuration
        dimension = embeddings.shape[1]
        # Use IndexHNSWFlat for better performance on larger datasets
        if len(all_chunks) > 1000:
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
            self.index.hnsw.efConstruction = 200
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Create TF-IDF index for hybrid search
        print("Creating TF-IDF index for hybrid search...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_chunks)
        
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        print("Indexing complete!")
    
    def process_documents_from_directory(self):
        """Process all documents from the documents directory"""
        documents = self.load_documents_from_directory()
        
        if not documents:
            print("No documents found to process!")
            return False
        
        self.process_documents(documents)
        return True
    
    def save_index(self):
        """Save index and metadata to disk"""
        print("Saving improved index...")
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_dir, "index.faiss"))
        
        # Save chunks and metadata
        with open(os.path.join(self.index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
            
        with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save TF-IDF components
        with open(os.path.join(self.index_dir, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        with open(os.path.join(self.index_dir, "tfidf_matrix.pkl"), "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        
        print("Improved index saved!")
    
    def load_index(self):
        """Load index and metadata from disk"""
        index_path = os.path.join(self.index_dir, "index.faiss")
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        tfidf_vectorizer_path = os.path.join(self.index_dir, "tfidf_vectorizer.pkl")
        tfidf_matrix_path = os.path.join(self.index_dir, "tfidf_matrix.pkl")
        
        required_files = [index_path, chunks_path, metadata_path]
        if not all(os.path.exists(p) for p in required_files):
            print("Index files not found. Please process documents first.")
            return False
        
        print("Loading improved index...")
        self.index = faiss.read_index(index_path)
        
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Load TF-IDF components if available
        if os.path.exists(tfidf_vectorizer_path) and os.path.exists(tfidf_matrix_path):
            with open(tfidf_vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open(tfidf_matrix_path, "rb") as f:
                self.tfidf_matrix = pickle.load(f)
            print("TF-IDF index loaded for hybrid search")
        
        print(f"Index loaded with {len(self.chunks)} chunks")
        return True
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword-based retrieval"""
        if self.index is None:
            raise ValueError("No index loaded. Please load or create an index first.")
        
        # Semantic search with embeddings
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        semantic_scores, semantic_indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
        
        # Keyword search with TF-IDF (if available)
        keyword_results = []
        if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            
            # Get top keyword matches
            keyword_indices = np.argsort(keyword_scores)[::-1][:top_k * 2]
            keyword_results = [(idx, keyword_scores[idx]) for idx in keyword_indices if keyword_scores[idx] > 0]
        
        # Combine and rerank results
        combined_results = {}
        
        # Add semantic results
        for score, idx in zip(semantic_scores[0], semantic_indices[0]):
            if idx < len(self.metadata):
                combined_results[idx] = {
                    'metadata': self.metadata[idx].copy(),
                    'semantic_score': float(score),
                    'keyword_score': 0.0,
                    'combined_score': float(score) * 0.7  # Weight semantic search higher
                }
        
        # Add keyword results
        for idx, score in keyword_results:
            if idx in combined_results:
                combined_results[idx]['keyword_score'] = score
                combined_results[idx]['combined_score'] = combined_results[idx]['semantic_score'] * 0.7 + score * 0.3
            else:
                if idx < len(self.metadata):
                    combined_results[idx] = {
                        'metadata': self.metadata[idx].copy(),
                        'semantic_score': 0.0,
                        'keyword_score': score,
                        'combined_score': score * 0.3
                    }
        
        # Sort by combined score and return top_k
        sorted_results = sorted(combined_results.values(), key=lambda x: x['combined_score'], reverse=True)[:top_k]
        
        # Format results
        results = []
        for result in sorted_results:
            metadata = result['metadata']
            metadata['semantic_score'] = result['semantic_score']
            metadata['keyword_score'] = result['keyword_score']
            metadata['combined_score'] = result['combined_score']
            results.append(metadata)
        
        return results
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:  # REDUCED top_k default
        """Search for relevant chunks using hybrid approach"""
        return self.hybrid_search(query, top_k)
    
    def query_llama(self, prompt: str, max_tokens: int = 300) -> str:  # REDUCED max_tokens
        """Query the local LLaMA server with improved error handling and proper parsing"""
        try:
            response = requests.post(
                f"{self.llama_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["</s>", "Human:", "User:", "\n\nQuestion:", "\n\nContext:"]
                },
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                # llama.cpp usually returns {"content": [{"text": "..."}]}
                if "content" in data and isinstance(data["content"], list):
                    content = "".join([c.get("text", "") for c in data["content"]]).strip()
                else:
                    content = str(data.get("content", "")).strip()

                # Clean up the response
                content = re.sub(r'\n+', '\n', content)
                return content if content else "No response from model."
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            return f"Error connecting to LLaMA server: {e}"

    def truncate_context_to_fit(self, context: str, question: str) -> str:
        """Truncate context to fit within token limits"""
        # Reserve tokens for prompt structure and answer
        base_prompt = f"""Context:

Question: {question}

Instructions:
- Answer based only on the provided context
- Be concise but comprehensive

Answer:"""
        
        base_tokens = self.estimate_tokens(base_prompt)
        available_context_tokens = self.max_context_tokens - base_tokens - 50  # Safety margin
        
        if available_context_tokens <= 0:
            return "Context too long for current model."
        
        context_tokens = self.estimate_tokens(context)
        
        if context_tokens <= available_context_tokens:
            return context
        
        # Truncate context by sentences to maintain readability
        sentences = re.split(r'(?<=[.!?])\s+', context)
        truncated_context = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            if current_tokens + sentence_tokens > available_context_tokens:
                break
            truncated_context += sentence + " "
            current_tokens += sentence_tokens
        
        return truncated_context.strip() + "\n\n[Context truncated due to length limits]"
    
    def format_sources_with_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources with better attribution and context"""
        formatted_sources = []
        
        for i, result in enumerate(results):
            source_info = {
                "source_id": i + 1,
                "filename": result["filename"],
                "title": result["title"],
                "relevance_score": round(result.get("combined_score", result.get("score", 0)), 3),
                "semantic_score": round(result.get("semantic_score", 0), 3),
                "keyword_score": round(result.get("keyword_score", 0), 3),
                "chunk_position": f"{result['chunk_id'] + 1} of {result.get('total_chunks', '?')}",
                "excerpt": result["text"][:150] + "..." if len(result["text"]) > 150 else result["text"],  # REDUCED excerpt
                "word_count": result.get("word_count", len(result["text"].split()))
            }
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    def answer_question(self, question: str, top_k: int = 3, include_context: bool = True) -> Dict[str, Any]:  # REDUCED top_k
        """Enhanced RAG pipeline with context size management"""
        start_time = time.time()
        
        # Step 1: Retrieve relevant chunks
        results = self.search(question, top_k=top_k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context_used": 0,
                "retrieval_time": time.time() - start_time,
                "generation_time": 0
            }
        
        # Step 2: Build context with size management
        context_pieces = []
        for i, result in enumerate(results):
            source_label = f"[Source {i+1}: {result['filename']}]"
            # Use shorter excerpts
            excerpt = result['text']
            if len(excerpt) > 400:  # Limit individual chunk size
                excerpt = excerpt[:400] + "..."
            context_pieces.append(f"{source_label}\n{excerpt}")
        
        raw_context = "\n\n".join(context_pieces)
        
        # Step 3: Truncate context if needed
        context = self.truncate_context_to_fit(raw_context, question)
        
        # Step 4: Create optimized prompt
        prompt = f"""Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- Be concise but comprehensive

Answer:"""
        
        # Debug: Check final prompt size
        prompt_tokens = self.estimate_tokens(prompt)
        print(f"Prompt tokens: {prompt_tokens}")
        
        retrieval_time = time.time() - start_time
        generation_start = time.time()
        
        # Step 5: Get LLM response
        answer = self.query_llama(prompt, max_tokens=300)  # REDUCED
        
        generation_time = time.time() - generation_start
        
        # Step 6: Format sources with detailed information
        formatted_sources = self.format_sources_with_context(results)
        
        return {
            "answer": answer,
            "sources": formatted_sources,
            "context_used": len(results),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(time.time() - start_time, 3),
            "raw_context": raw_context if include_context else None,
            "prompt_tokens": prompt_tokens,  # NEW: Token debugging
            "context_truncated": context != raw_context  # NEW: Truncation indicator
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.chunks:
            return {"error": "No index loaded"}
        
        # Calculate document statistics
        doc_stats = {}
        for metadata in self.metadata:
            filename = metadata["filename"]
            if filename not in doc_stats:
                doc_stats[filename] = {"chunks": 0, "total_words": 0}
            doc_stats[filename]["chunks"] += 1
            doc_stats[filename]["total_words"] += metadata.get("word_count", 0)
        
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(doc_stats),
            "average_chunk_size": sum(len(chunk.split()) for chunk in self.chunks) / len(self.chunks),
            "max_context_tokens": self.max_context_tokens,  # NEW
            "index_type": type(self.index).__name__,
            "has_tfidf": self.tfidf_vectorizer is not None,
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
            "documents": doc_stats
        }
