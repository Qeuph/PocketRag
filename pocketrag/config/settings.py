"""
PocketRAG - Production Configuration Module
"""
import os
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union


@dataclass
class Config:
    """Application configuration with sensible defaults."""
    
    # Database settings
    db_path: str = "./.pocketrag/data"
    table_name: str = "documents"
    
    # Embedding model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_by_sentence: bool = True  # Try to break on sentence boundaries
    
    # Search settings
    default_top_k: int = 5
    score_threshold: float = 0.0  # Minimum similarity score threshold
    enable_hybrid_search: bool = False  # Enable BM25 + vector search
    
    # LLM settings
    default_model: str = "qwen3.5:0.8b"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Re-ranking settings
    enable_reranking: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Supported file extensions
    supported_extensions: tuple = field(default_factory=lambda: (
        '.pdf', '.txt', '.md', '.py', '.js', '.json',
        '.ts', '.tsx', '.html', '.css', '.java', '.cpp',
        '.c', '.h', '.go', '.rs', '.rb', '.php',
        '.docx', '.csv', '.xml', '.yaml', '.yml'
    ))
    
    @property
    def db_dir(self) -> Path:
        """Get the database directory path."""
        return Path(self.db_path).parent
    
    def ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "db_path": str(self.db_path),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_model": self.default_model,
            "enable_hybrid_search": self.enable_hybrid_search,
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to a JSON file."""
        if path is None:
            path = self.db_dir / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def load(self, path: Optional[Path] = None) -> None:
        """Load config from a JSON file."""
        if path is None:
            path = self.db_dir / "config.json"

        if not path.exists():
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    # Only set attributes that exist and are not properties
                    if hasattr(self, key) and not isinstance(getattr(type(self), key, None), property):
                        setattr(self, key, value)
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    def setup_logging(self) -> None:
        """Configure logging for the application."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        handlers: List[logging.Handler] = [logging.StreamHandler()]
        
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.log_file))
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format=log_format,
            handlers=handlers
        )


# Global config instance
config = Config()
