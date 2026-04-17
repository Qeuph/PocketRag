"""
PocketRAG - Document Processing Utilities
"""
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading and extracting text from various document formats."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': '_load_pdf',
        '.txt': '_load_text',
        '.md': '_load_text',
        '.py': '_load_text',
        '.js': '_load_text',
        '.json': '_load_text',
        '.ts': '_load_text',
        '.tsx': '_load_text',
        '.html': '_load_text',
        '.css': '_load_text',
        '.java': '_load_text',
        '.cpp': '_load_text',
        '.c': '_load_text',
        '.h': '_load_text',
        '.go': '_load_text',
        '.rs': '_load_text',
        '.rb': '_load_text',
        '.php': '_load_text',
    }
    
    def __init__(self):
        self._pdf_reader = None
    
    def _get_pdf_reader(self):
        """Lazy load PDF reader to avoid import overhead when not needed."""
        if self._pdf_reader is None:
            try:
                from pypdf import PdfReader
                self._pdf_reader = PdfReader
            except ImportError:
                logger.warning("pypdf not installed. PDF support disabled.")
                return None
        return self._pdf_reader
    
    def load(self, file_path: Path) -> Optional[str]:
        """
        Extract text content from a file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Extracted text content or None if loading failed
        """
        if not file_path.exists():
            logger.debug(f"File not found: {file_path}")
            return None
        
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.debug(f"Unsupported file type: {ext}")
            return None
        
        method_name = self.SUPPORTED_EXTENSIONS[ext]
        method = getattr(self, method_name)
        
        try:
            return method(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            return None
    
    def _load_pdf(self, file_path: Path) -> Optional[str]:
        """Extract text from PDF files."""
        PdfReader = self._get_pdf_reader()
        if PdfReader is None:
            return None
        
        try:
            reader = PdfReader(file_path)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n".join(pages_text) if pages_text else None
        except Exception as e:
            logger.warning(f"PDF extraction failed for {file_path}: {e}")
            return None
    
    def _load_text(self, file_path: Path) -> Optional[str]:
        """Load plain text files."""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                logger.warning(f"Text decoding failed for {file_path}: {e}")
                return None
