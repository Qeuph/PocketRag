import lancedb
from sentence_transformers import SentenceTransformer

class Searcher:
    def __init__(self, db_path="./.pocketrag/data"):
        self.db_path = db_path
        # Connect to the existing DB
        try:
            self.db = lancedb.connect(self.db_path)
            self.table = self.db.open_table("documents")
        except Exception as e:
            print(f"❌ Error connecting to database. Did you run indexer.py first? ({e})")
            raise
        
        # Load the same model used in indexer.py
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def search(self, query: str, top_k: int = 3):
        """
        Performs a vector similarity search.
        Returns a list of dictionaries with 'text' and 'source'.
        """
        # 1. Convert user query to vector
        query_embedding = self.model.encode(query)

        # 2. Perform search in LanceDB
        # .search() handles the vector similarity calculation
        results = (
            self.table.search(query_embedding)
            .limit(top_k)
            .to_list()
        )
        
        return results

    def format_context(self, results):
        """
        Formats the search results into a clean string for the LLM.
        This is crucial for good RAG performance.
        """
        context_text = ""
        for i, res in enumerate(results):
            context_text += f"Source: {res['source']}\nContent: {res['text']}\n\n"
        return context_text

# Example Usage
if __name__ == "__main__":
    searcher = Searcher()
    query = "What is the primary goal of this project?"
    
    results = searcher.search(query)
    formatted = searcher.format_context(results)
    
    print(f"--- Context for LLM ---\n{formatted}")