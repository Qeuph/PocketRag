import os
import lancedb
import pyarrow as pa
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from tqdm import tqdm

class Indexer:
    def __init__(self, db_path="./.pocketrag/data"):
        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        # Using all-MiniLM-L6-v2 for the best balance of speed vs accuracy
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.table_name = "documents"

    def _get_file_content(self, file_path):
        """Extracts text from various file formats."""
        ext = file_path.suffix.lower()
        try:
            if ext == '.pdf':
                reader = PdfReader(file_path)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif ext in ['.txt', '.md', '.py', '.js', '.json']:
                return file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")
        return None

    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """Splits text into overlapping chunks to preserve context."""
        if not text: return []
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
        return chunks

    def index_directory(self, directory_path):
        """Reads a directory, chunks text, embeds, and saves to LanceDB."""
        path = Path(directory_path)
        files = list(path.rglob('*')) # Recursive search
        all_data = []

        print(f"🔍 Indexing {len(files)} files...")

        for file in tqdm(files, desc="Processing files"):
            content = self._get_file_content(file)
            if content:
                chunks = self._chunk_text(content)
                for chunk in chunks:
                    all_data.append({
                        "text": chunk,
                        "source": str(file)
                    })

        if not all_data:
            print("No indexable content found.")
            return

        print(f"🧠 Generating embeddings for {len(all_data)} chunks...")
        
        # Batch embedding generation (much faster than looping)
        texts = [d["text"] for d in all_data]
        embeddings = self.model.encode(texts)

        # Prepare for LanceDB
        data_to_store = []
        for i in range(len(all_data)):
            data_to_store.append({
                "vector": embeddings[i],
                "text": all_data[i]["text"],
                "source": all_data[i]["source"]
            })

        # Create/Update table
        # LanceDB schema is inferred from the first record
        table = self.db.create_table(self.table_name, data=data_to_store, mode="overwrite")
        print(f"✅ Successfully indexed {len(all_data)} chunks to {self.db_path}")

# Example Usage
if __name__ == "__main__":
    indexer = Indexer()
    # Replace with a real folder path
    indexer.index_directory("./my_docs")