"""
PocketRAG - Document Indexer Module
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from pocketrag.config import config
from pocketrag.core.chunker import TextChunker
from pocketrag.core.embedding import EmbeddingEngine
from pocketrag.core.vector_store import VectorStore
from pocketrag.utils.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class Document:
    """Represents a document chunk for indexing."""
    
    def __init__(self, text: str, source: str, metadata: Optional[Dict] = None):
        self.text = text
        self.source = source
        self.metadata = metadata or {}


class Indexer:
    """
    Handles document indexing pipeline:
    Load -> Chunk -> Embed -> Store
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the indexer.
        
        Args:
            db_path: Path to the vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of the embedding model
        """
        self.db_path = db_path or config.db_path
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.embedding_model = embedding_model or config.embedding_model
        
        # Initialize components
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.chunk_size, self.chunk_overlap)
        self.embedder = EmbeddingEngine(self.embedding_model)
        self.store = VectorStore(self.db_path, config.table_name)
        
        logger.info(f"Indexer initialized with DB at {self.db_path}")
    
    def index_directory(
        self,
        directory_path: str,
        recursive: bool = True
    ) -> Dict[str, int]:
        """
        Index all documents in a directory.
        
        Args:
            directory_path: Path to the directory to index
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with indexing statistics
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        # Find all files
        pattern = "**/*" if recursive else "*"
        files = list(path.glob(pattern))
        files = [f for f in files if f.is_file()]
        
        logger.info(f"Found {len(files)} files to process")
        
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "errors": 0
        }
        
        documents = []
        
        # Process files
        for file_path in tqdm(files, desc="Processing files"):
            try:
                content = self.loader.load(file_path)
                
                if content is None:
                    stats["files_skipped"] += 1
                    continue
                
                # Chunk the content
                chunks = self.chunker.chunk(content)
                
                if not chunks:
                    stats["files_skipped"] += 1
                    continue
                
                # Create document objects
                for chunk in chunks:
                    documents.append(Document(
                        text=chunk,
                        source=str(file_path)
                    ))
                
                stats["files_processed"] += 1
                stats["chunks_created"] += len(chunks)
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                stats["errors"] += 1
        
        if not documents:
            logger.warning("No documents to index")
            return stats
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        texts = [doc.text for doc in documents]
        
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
        
        # Prepare data for storage
        data_to_store = []
        for i, doc in enumerate(documents):
            data_to_store.append({
                "vector": embeddings[i].tolist(),
                "text": doc.text,
                "source": doc.source
            })
        
        # Store in vector database
        self.store.insert(data_to_store)
        
        logger.info(
            f"Indexing complete: {stats['files_processed']} files, "
            f"{stats['chunks_created']} chunks"
        )
        
        return stats
    
    def index_file(self, file_path: str) -> Dict[str, int]:
        """
        Index a single file.
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            Dictionary with indexing statistics
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        # Load content
        content = self.loader.load(path)
        
        if content is None:
            return {"files_processed": 0, "chunks_created": 0}
        
        # Chunk
        chunks = self.chunker.chunk(content)
        
        if not chunks:
            return {"files_processed": 0, "chunks_created": 0}
        
        # Generate embeddings
        embeddings = self.embedder.embed(chunks)
        
        # Prepare data
        data_to_store = []
        for i, chunk in enumerate(chunks):
            data_to_store.append({
                "vector": embeddings[i].tolist(),
                "text": chunk,
                "source": str(path)
            })
        
        # Store
        self.store.insert(data_to_store)
        
        return {"files_processed": 1, "chunks_created": len(chunks)}
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.store.clear()
        logger.info("Cleared all indexed documents")
    
    def count(self) -> int:
        """Get the number of indexed chunks."""
        return self.store.count()
