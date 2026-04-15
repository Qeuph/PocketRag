import typer
from rich.console import Console
from engine.indexer import Indexer
from engine.chat import ChatEngine
from pathlib import Path

app = typer.Typer(
    name="pocketrag",
    help="⚡ PocketRAG: Lightning fast, local AI document engine.",
    add_completion=False,
)

console = Console()

@app.command()
def init():
    """Initialize the PocketRAG database."""
    console.print("[bold blue]🚀 Initializing PocketRAG...[/bold blue]")
    try:
        Indexer() # This will create the .pocketrag directory
        console.print("[green]✅ Database ready![/green]")
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")

@app.command()
def add(path: Path = typer.Argument(..., help="Path to the directory containing your documents.")):
    """Add a directory of documents to the index."""
    if not path.exists():
        console.print(f"[red]❌ Directory not found: {path}[/red]")
        raise typer.Exit()
    
    indexer = Indexer()
    console.print(f"[bold green]📚 Indexing documents from: {path}[/bold green]")
    indexer.index_directory(str(path))
    console.print("[bold green]✅ Indexing complete![/bold green]")

@app.command()
def chat(model: str = typer.Option("qwen3.5:0.8b", help="The Ollama model to use.")):
    """Start an interactive chat session."""
    engine = ChatEngine(model_name=model)
    console.print(f"[bold cyan]🤖 Chatting with {model}. Type 'quit' to exit.[/bold cyan]")
    
    while True:
        try:
            query = typer.prompt("You")
            if query.lower() in ["quit", "exit"]:
                break
            engine.chat(query)
        except typer.Abort:
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

if __name__ == "__main__":
    app()