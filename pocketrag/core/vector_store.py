"""
PocketRAG - Vector Store Module using LanceDB
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using LanceDB."""
    
    def __init__(self, db_path: str, table_name: str = "documents"):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to the LanceDB database directory
            table_name: Name of the table to use
        """
        self.db_path = db_path
        self.table_name = table_name
        self._db = None
        self._table = None
    
    @property
    def db(self):
        """Lazy load the LanceDB connection."""
        if self._db is None:
            try:
                import lancedb
                Path(self.db_path).mkdir(parents=True, exist_ok=True)
                self._db = lancedb.connect(self.db_path)
                logger.info(f"Connected to LanceDB at {self.db_path}")
            except ImportError:
                logger.error("lancedb not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to connect to LanceDB: {e}")
                raise
        return self._db
    
    @property
    def table(self):
        """Get or create the documents table."""
        if self._table is None:
            try:
                # Try to open existing table
                self._table = self.db.open_table(self.table_name)
                logger.info(f"Opened existing table: {self.table_name}")
            except Exception:
                # Table doesn't exist yet - it will be created on first insert
                logger.info(f"Table {self.table_name} will be created on first insert")
        return self._table
    
    def insert(self, documents: List[Dict[str, Any]]) -> int:
        """
        Insert documents into the vector store.
        
        Args:
            documents: List of dicts with 'vector', 'text', and 'source' keys
            
        Returns:
            Number of documents inserted
        """
        if not documents:
            return 0
        
        try:
            # Create or overwrite table with new data
            self._table = self.db.create_table(
                self.table_name,
                data=documents,
                mode="overwrite"
            )
            logger.info(f"Inserted {len(documents)} documents into {self.table_name}")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
    
    def search(
        self,
        query_vector: Any,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        if self._table is None:
            logger.warning("No table available for search")
            return []
        
        try:
            results = (
                self.table.search(query_vector)
                .limit(top_k)
                .to_list()
            )
            logger.debug(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def exists(self) -> bool:
        """Check if the table exists and has data."""
        try:
            tbl = self.db.open_table(self.table_name)
            return True
        except Exception:
            return False
    
    def count(self) -> int:
        """Get the number of documents in the store."""
        if not self.exists():
            return 0
        try:
            return len(self.table.to_list())
        except Exception:
            return 0
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        try:
            self.db.drop_table(self.table_name)
            self._table = None
            logger.info(f"Cleared table: {self.table_name}")
        except Exception as e:
            logger.warning(f"Failed to clear table: {e}")
