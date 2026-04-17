"""
PocketRAG - Production Configuration Module
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


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
    
    # Search settings
    default_top_k: int = 5
    
    # LLM settings
    default_model: str = "qwen3.5:0.8b"
    
    # Supported file extensions
    supported_extensions: tuple = (
        '.pdf', '.txt', '.md', '.py', '.js', '.json',
        '.ts', '.tsx', '.html', '.css', '.java', '.cpp',
        '.c', '.h', '.go', '.rs', '.rb', '.php'
    )
    
    @property
    def db_dir(self) -> Path:
        """Get the database directory path."""
        return Path(self.db_path).parent
    
    def ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
