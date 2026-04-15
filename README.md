
# ⚡ PocketRAG

**Privacy-first, local RAG engine. Run LLM search on your documents without the cloud.**

PocketRAG turns your local documents into an intelligent, searchable database. Built for speed, privacy, and minimal resource usage (optimized for Qwen3.5 0.8B).

![PixelArt](assets/lr.jpeg)

## 🚀 Why PocketRAG?
- **100% Offline:** Your data never leaves your machine.
- **Ultra-Light:** Designed to run on consumer laptops (no massive GPUs required).
- **Zero-Setup:** Single-command CLI.
- **Smart Retrieval:** Uses vector embeddings to find exact answers, not just keywords.

## 🛠️ Quick Start

```bash
# 1. Install
git clone https://github.com/YOUR_USERNAME/pocketrag
cd pocketrag
pip install -e .

# 2. Initialize and Add Docs
pocketrag init
pocketrag add ./my_documents

# 3. Chat
pocketrag chat
```

## 🏗️ Architecture
PocketRAG uses:

LanceDB: Serverless vector storage.

Ollama: Local LLM inference.

SentenceTransformers: High-performance local embeddings.

Rich/Typer: For a beautiful CLI experience.

## 🤝 Contributing

Built for developers who value privacy. PRs are welcome!

If this project saved you time, please ⭐ star the repo!
