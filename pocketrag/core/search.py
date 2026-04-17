"""
PocketRAG - Document Search Module
"""
import logging
from typing import List, Dict, Any, Optional

from pocketrag.config import config
from pocketrag.core.embedding import EmbeddingEngine
from pocketrag.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result."""
    
    def __init__(
        self,
        text: str,
        source: str,
        score: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        self.text = text
        self.source = source
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"SearchResult(source={self.source}, score={self.score:.4f})"


class Searcher:
    """
    Handles document search and retrieval.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the searcher.
        
        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model
        """
        self.db_path = db_path or config.db_path
        self.embedding_model = embedding_model or config.embedding_model
        
        # Initialize components
        self.embedder = EmbeddingEngine(self.embedding_model)
        self.store = VectorStore(self.db_path, config.table_name)
        
        logger.info(f"Searcher initialized with DB at {self.db_path}")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.store.exists():
            logger.warning("No indexed documents found. Run 'pocketrag add' first.")
            return []
        
        # Generate query embedding
        query_vector = self.embedder.embed_single(query)
        
        if len(query_vector) == 0:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search the vector store
        raw_results = self.store.search(query_vector.tolist(), top_k=top_k)
        
        # Convert to SearchResult objects
        results = []
        for res in raw_results:
            results.append(SearchResult(
                text=res.get('text', ''),
                source=res.get('source', ''),
                score=res.get('_distance', 0.0)
            ))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def format_context(
        self,
        results: List[SearchResult],
        include_sources: bool = True
    ) -> str:
        """
        Format search results into context text for LLM prompts.
        
        Args:
            results: List of search results
            include_sources: Whether to include source information
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        formatted_parts = []
        for i, result in enumerate(results, 1):
            parts = [f"[Document {i}]"]
            if include_sources:
                parts.append(f"Source: {result.source}")
            parts.append(f"Content: {result.text}")
            formatted_parts.append("\n".join(parts))
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 5
    ) -> tuple[List[SearchResult], str]:
        """
        Search and return both results and formatted context.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            Tuple of (results list, formatted context string)
        """
        results = self.search(query, top_k)
        context = self.format_context(results)
        return results, context
    
    def count(self) -> int:
        """Get the number of indexed documents."""
        return self.store.count()
    
    def is_indexed(self) -> bool:
        """Check if documents have been indexed."""
        return self.store.exists() and self.count() > 0
