import asyncio
import os
import json
from typing import Dict, Any
from pyhypercycle_aim import SimpleServer, aim_uri, JSONResponseCORS
from improved_rag_system import ImprovedRAG

class RAGAIMServer(SimpleServer):
    manifest = {
        "name": "HyperCycle-FAQ",
        "short_name": "hypercyclefaq",
        "version": "1.0.0",
        "documentation_url": "",
        "license": "MIT",
        "terms_of_service": "",
        "author": "ants"
    }
    
    def __init__(self):
        super().__init__()
        self.rag = None
        self.initialize_rag()
    
    def initialize_rag(self):
        """Initialize the RAG system with context management"""
        print("Initializing RAG system with context management...")
        
        # Get LLaMA URL from environment or use default
        llama_url = os.getenv("LLAMA_URL", "http://localhost:8080")
        
        # Get context limit from environment or use conservative default
        max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "1200"))
        
        print(f"Using LLaMA URL: {llama_url}")
        print(f"Max context tokens: {max_context_tokens}")
        
        self.rag = ImprovedRAG(
            llama_url=llama_url,
            index_dir="/app/rag_index",
            documents_dir="/app/documents",
            max_context_tokens=max_context_tokens  
        )
        
        # Try to load existing index, otherwise process documents
        if not self.rag.load_index():
            print("No existing index found. Processing documents...")
            if self.rag.process_documents_from_directory():
                self.rag.save_index()
                print("Index created and saved!")
            else:
                print("Warning: No documents found to process!")
        else:
            # Show stats after loading
            stats = self.rag.get_system_stats()
            print(f"Loaded index with {stats.get('total_chunks', 0)} chunks from {stats.get('total_documents', 0)} documents")
        
        print("RAG system initialized successfully!")
    
    @aim_uri(
        uri="/query",
        methods=["POST"],
        endpoint_manifest={
            "input_query": {},
            "input_body": {
                "question": "<string>",
                "top_k": "<int, optional, default=3>",  
                "include_context": "<bool, optional, default=false>"
            },
            "output": {
                "answer": "<string>",
                "sources": "<array>",
                "context_used": "<int>",
                "retrieval_time": "<float>",
                "generation_time": "<float>",
                "total_time": "<float>",
                "prompt_tokens": "<int>",
                "context_truncated": "<bool>"
            },
            "documentation": "Query the RAG system with a question about HyperCycle and get an answer with sources",
            "example_calls": [{
                "method": "POST",
                "body": {
                    "question": "What is the authentication process?",
                    "top_k": 3
                },
                "output": {
                    "answer": "Authentication is done via JWT tokens...",
                    "sources": [{"filename": "auth.txt", "relevance_score": 0.95}],
                    "context_used": 1,
                    "retrieval_time": 0.1,
                    "generation_time": 2.3,
                    "total_time": 2.4,
                    "prompt_tokens": 850,
                    "context_truncated": False
                }
            }],
        }
    )
    async def query_endpoint(self, request):
        """Handle RAG queries with context size management"""
        try:
            body = await request.json()
            question = body.get("question", "")
            top_k = body.get("top_k", 3)  # REDUCED default from 5 to 3
            include_context = body.get("include_context", False)
            
            if not question:
                return JSONResponseCORS({
                    "error": "Question is required"
                }, status_code=400)
            
            if self.rag is None:
                return JSONResponseCORS({
                    "error": "RAG system not initialized"
                }, status_code=500)
            
            # Validate top_k to prevent context overflow
            if top_k > 10:
                print(f"Warning: top_k={top_k} is very high, reducing to 10")
                top_k = 10
            
            print(f"Processing query: '{question[:50]}...' with top_k={top_k}")
            
            # Run the RAG query in a thread pool to avoid blocking
            result = await asyncio.to_thread(
                self.rag.answer_question,
                question,
                top_k,
                include_context
            )
            
            # Log token usage for debugging
            if "prompt_tokens" in result:
                print(f"Query completed - Prompt tokens: {result['prompt_tokens']}, "
                      f"Context truncated: {result.get('context_truncated', False)}")
            
            return JSONResponseCORS({"answer": result["answer"]})
            
        except Exception as e:
            print(f"Query error: {str(e)}")
            return JSONResponseCORS({
                "error": f"Query failed: {str(e)}"
            }, status_code=500)
    
    
    @aim_uri(
        uri="/reindex",
        methods=["POST"],
        endpoint_manifest={
            "input_query": {},
            "input_body": {},
            "output": {
                "success": "<bool>",
                "message": "<string>",
                "stats": "<object>"
            },
            "documentation": "Reprocess all documents and rebuild the index",
            "example_calls": [{
                "method": "POST",
                "body": {},
                "output": {
                    "success": True,
                    "message": "Index rebuilt successfully",
                    "stats": {"total_chunks": 15, "total_documents": 3}
                }
            }],
            
            "is_public": True
        }
    )
    async def reindex_endpoint(self, request):
        """Rebuild the RAG index"""
        try:
            if self.rag is None:
                return JSONResponseCORS({
                    "error": "RAG system not initialized"
                }, status_code=500)
            
            print("Starting reindex process...")
            
            # Run reindexing in a thread pool
            success = await asyncio.to_thread(
                self.rag.process_documents_from_directory
            )
            
            if success:
                await asyncio.to_thread(self.rag.save_index)
                stats = await asyncio.to_thread(self.rag.get_system_stats)
                
                print(f"Reindex completed: {stats.get('total_chunks', 0)} chunks, "
                      f"{stats.get('total_documents', 0)} documents")
                
                return JSONResponseCORS({
                    "success": True,
                    "message": "Index rebuilt successfully",
                    "stats": stats
                })
            else:
                return JSONResponseCORS({
                    "success": False,
                    "message": "No documents found to process"
                })
                
        except Exception as e:
            print(f"Reindex error: {str(e)}")
            return JSONResponseCORS({
                "error": f"Reindexing failed: {str(e)}"
            }, status_code=500)
    
    @aim_uri(
        uri="/stats",
        methods=["GET"],
        endpoint_manifest={
            "input_query": {},
            "input_body": {},
            "output": {
                "total_chunks": "<int>",
                "total_documents": "<int>",
                "average_chunk_size": "<float>",
                "max_context_tokens": "<int>",  # NEW
                "index_type": "<string>",
                "has_tfidf": "<bool>",
                "embedding_model": "<int>",
                "documents": "<object>"
            },
            "documentation": "Get system statistics and information about the RAG index",
            "example_calls": [{
                "method": "GET",
                "body": {},
                "output": {
                    "total_chunks": 15,
                    "total_documents": 3,
                    "average_chunk_size": 45.2,
                    "max_context_tokens": 1200,
                    "index_type": "IndexFlatIP",
                    "has_tfidf": True,
                    "embedding_model": 384,
                    "documents": {"auth.txt": {"chunks": 1, "total_words": 15}}
                }
            }],
            "is_public": True
        }
    )
    async def stats_endpoint(self, request):
        """Get RAG system statistics"""
        try:
            if self.rag is None:
                return JSONResponseCORS({
                    "error": "RAG system not initialized"
                }, status_code=500)
            
            stats = await asyncio.to_thread(self.rag.get_system_stats)
            return JSONResponseCORS(stats)
            
        except Exception as e:
            print(f"Stats error: {str(e)}")
            return JSONResponseCORS({
                "error": f"Failed to get stats: {str(e)}"
            }, status_code=500)
    
    
    
    
    @aim_uri(
        uri="/health",
        methods=["GET"],
        endpoint_manifest={
            "input_query": {},
            "input_body": {},
            "output": {
                "status": "<string>",
                "rag_initialized": "<bool>",
                "llama_url": "<string>",
                "max_context_tokens": "<int>"  # NEW
            },
            "documentation": "Health check endpoint",
            "example_calls": [{
                "method": "GET",
                "body": {},
                "output": {
                    "status": "healthy",
                    "rag_initialized": True,
                    "llama_url": "http://llama-server:8080",
                    "max_context_tokens": 1200
                }
            }],
            "is_public": True
        }
    )
    async def health_endpoint(self, request):
        """Health check with context info"""
        return JSONResponseCORS({
            "status": "healthy",
            "rag_initialized": self.rag is not None,
            "llama_url": os.getenv("LLAMA_URL", "http://localhost:8080"),
            "max_context_tokens": self.rag.max_context_tokens if self.rag else 0
        })

if __name__ == '__main__':
    server = RAGAIMServer()
    server.run(uvicorn_kwargs={"port": 4000, "host": "0.0.0.0"})
