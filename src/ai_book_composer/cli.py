"""Command-line interface for AI Book Composer."""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from .workflow import BookComposerWorkflow
from .config import settings


console = Console()


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing source files"
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory for output files"
)
@click.option(
    "--title",
    "-t",
    default="Composed Book",
    help="Book title"
)
@click.option(
    "--author",
    "-a",
    default="AI Book Composer",
    help="Book author"
)
@click.option(
    "--language",
    "-l",
    default="en-US",
    help="Target language (e.g., en-US, es-ES, fr-FR)"
)
@click.option(
    "--max-iterations",
    default=3,
    help="Maximum revision iterations"
)
def main(
    input_dir: str,
    output_dir: str,
    title: str,
    author: str,
    language: str,
    max_iterations: int
):
    """AI Book Composer - Generate comprehensive books from source files.
    
    This tool uses Deep-Agent architecture with LangGraph to:
    1. Plan the book structure
    2. Execute content generation
    3. Critique and iterate for quality
    """
    console.print(Panel.fit(
        "[bold blue]AI Book Composer[/bold blue]\n"
        "Using Deep-Agent pattern to compose books",
        border_style="blue"
    ))
    
    # Display configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Input Directory", input_dir)
    config_table.add_row("Output Directory", output_dir)
    config_table.add_row("Book Title", title)
    config_table.add_row("Author", author)
    config_table.add_row("Language", language)
    config_table.add_row("LLM Provider", settings.llm_provider)
    config_table.add_row("LLM Model", settings.llm_model)
    config_table.add_row("Max Iterations", str(max_iterations))
    
    console.print(config_table)
    console.print()
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize workflow
    try:
        console.print("[yellow]Initializing workflow...[/yellow]")
        workflow = BookComposerWorkflow(
            input_directory=input_dir,
            output_directory=output_dir,
            language=language,
            book_title=title,
            book_author=author,
            max_iterations=max_iterations
        )
        
        # Run workflow
        console.print("[yellow]Starting book composition...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Composing book...", total=None)
            
            final_state = workflow.run()
            
            progress.update(task, completed=True)
        
        # Display results
        console.print()
        console.print(Panel.fit(
            "[bold green]✓ Book composition completed![/bold green]",
            border_style="green"
        ))
        
        # Show results
        results_table = Table(title="Results", show_header=False)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        results_table.add_row("Status", final_state.get("status", "unknown"))
        results_table.add_row("Chapters Generated", str(len(final_state.get("chapters", []))))
        results_table.add_row("References", str(len(final_state.get("references", []))))
        results_table.add_row("Iterations", str(final_state.get("iterations", 0)))
        
        quality_score = final_state.get("quality_score")
        if quality_score is not None:
            results_table.add_row("Quality Score", f"{quality_score:.2f}")
        
        output_path = final_state.get("final_output_path")
        if output_path:
            results_table.add_row("Output File", output_path)
        
        console.print(results_table)
        
        # Show feedback if any
        feedback = final_state.get("critic_feedback")
        if feedback:
            console.print()
            console.print(Panel(
                feedback,
                title="Critic Feedback",
                border_style="yellow"
            ))
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise click.Abort()


if __name__ == "__main__":
    main()
