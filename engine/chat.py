import ollama
from search import Searcher

class ChatEngine:
    def __init__(self, model_name="qwen3.5:0.8b"):
        self.searcher = Searcher()
        self.model_name = model_name

    def chat(self, user_input: str):
        print("\n🔍 Searching documents...")
        
        # 1. Retrieve relevant context
        results = self.searcher.search(user_input, top_k=3)
        context = self.searcher.format_context(results)

        # 2. Construct the Prompt
        # Small models (0.8B) work best with very clear, specific instructions
        system_prompt = (
            "You are a helpful local assistant. Use the provided context to answer the question. "
            "If the answer is not in the context, say so. Do not make up information."
        )
        
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}"

        print("🤖 Thinking...\n")

        try:
            # 3. Stream the response from Ollama
            stream = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                stream=True,
            )

            # Print as it generates (Streaming)
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print("\n")

        except Exception as e:
            print(f"\n❌ Connection Error: Is Ollama running? (Error: {e})")

# Example Usage
if __name__ == "__main__":
    engine = ChatEngine()
    print("PocketRAG Ready. Type 'quit' to exit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["quit", "exit"]:
            break
        engine.chat(user_query)