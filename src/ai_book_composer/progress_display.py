"""Progress display utilities for showing execution steps, thoughts, and actions."""

from typing import Optional, Any, Dict, List
from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text


# Global console instance
console = Console()


class ProgressDisplay:
    """Manages the display of execution progress, thoughts, and actions."""
    
    def __init__(self):
        self.console = Console()
        self._current_phase = None
        self._live = None
    
    def show_phase(self, phase_name: str, description: str, style: str = "bold cyan"):
        """Display the current execution phase.
        
        Args:
            phase_name: Name of the phase (e.g., "Planning", "Execution")
            description: Description of what the phase does
            style: Rich style for the phase name
        """
        self._current_phase = phase_name
        self.console.print()
        self.console.print(Panel(
            f"[{style}]{phase_name}[/{style}]\n{description}",
            border_style=style.split()[-1] if " " in style else style,
            title=f"⚙️  Phase: {phase_name}",
            title_align="left"
        ))
    
    def show_thought(self, thought: str, emoji: str = "💭"):
        """Display an agent's thought process.
        
        Args:
            thought: The thought/reasoning text
            emoji: Emoji to use as prefix
        """
        self.console.print(f"{emoji} [italic yellow]Thought:[/italic yellow] {thought}")
    
    def show_action(self, action: str, emoji: str = "🔧"):
        """Display an action being taken.
        
        Args:
            action: Description of the action
            emoji: Emoji to use as prefix
        """
        self.console.print(f"{emoji} [bold green]Action:[/bold green] {action}")
    
    def show_observation(self, observation: str, emoji: str = "👁️"):
        """Display an observation or result.
        
        Args:
            observation: The observation text
            emoji: Emoji to use as prefix
        """
        self.console.print(f"{emoji} [blue]Observation:[/blue] {observation}")
    
    def show_step(self, step_num: int, total_steps: int, description: str):
        """Display a step in a multi-step process.
        
        Args:
            step_num: Current step number
            total_steps: Total number of steps
            description: Description of the step
        """
        progress_bar = "█" * step_num + "░" * (total_steps - step_num)
        self.console.print(
            f"📊 [cyan]Step {step_num}/{total_steps}[/cyan] {progress_bar} - {description}"
        )
    
    def show_task(self, task_name: str, status: str = "started"):
        """Display a task being executed.
        
        Args:
            task_name: Name of the task
            status: Status of the task (started, completed, failed)
        """
        emoji_map = {
            "started": "▶️",
            "completed": "✅",
            "failed": "❌",
            "in_progress": "⏳"
        }
        color_map = {
            "started": "yellow",
            "completed": "green",
            "failed": "red",
            "in_progress": "cyan"
        }
        
        emoji = emoji_map.get(status, "➡️")
        color = color_map.get(status, "white")
        self.console.print(f"{emoji} [{color}]Task: {task_name}[/{color}]")
    
    def show_plan(self, plan: List[Dict[str, Any]]):
        """Display a structured plan.
        
        Args:
            plan: List of plan items/tasks
        """
        self.console.print()
        tree = Tree("📋 [bold]Generated Plan[/bold]")
        
        for i, task in enumerate(plan, 1):
            task_name = task.get("task", "Unknown Task")
            description = task.get("description", "")
            status = task.get("status", "pending")
            
            status_emoji = "⏳" if status == "pending" else "✓" if status == "completed" else "▶️"
            task_node = tree.add(f"{status_emoji} Step {i}: [cyan]{task_name}[/cyan]")
            if description:
                task_node.add(f"[dim]{description}[/dim]")
        
        self.console.print(tree)
    
    def show_files(self, files: List[Dict[str, Any]], title: str = "Files to Process"):
        """Display a list of files.
        
        Args:
            files: List of file information dictionaries
            title: Title for the file list
        """
        table = Table(title=f"📁 {title}", show_header=True, header_style="bold cyan")
        table.add_column("File Name", style="white")
        table.add_column("Type", style="yellow")
        table.add_column("Size", justify="right", style="green")
        
        for file_info in files[:10]:  # Show first 10 files
            name = file_info.get("name", "unknown")
            extension = file_info.get("extension", "").lstrip(".")
            size = file_info.get("size", 0)
            
            # Format size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            
            table.add_row(name, extension.upper() if extension else "unknown", size_str)
        
        if len(files) > 10:
            table.add_row("...", "...", f"... and {len(files) - 10} more files")
        
        self.console.print(table)
    
    def show_chapter_info(self, chapter_num: int, title: str, status: str = "generating"):
        """Display information about a chapter being generated.
        
        Args:
            chapter_num: Chapter number
            title: Chapter title
            status: Status of chapter generation
        """
        emoji = "📝" if status == "generating" else "✅" if status == "completed" else "📖"
        self.console.print(f"{emoji} [bold]Chapter {chapter_num}:[/bold] [cyan]{title}[/cyan]")
    
    def show_critique_summary(self, quality_score: Optional[float], feedback: str):
        """Display critique feedback summary.
        
        Args:
            quality_score: Quality score (0-1)
            feedback: Feedback text
        """
        self.console.print()
        
        if quality_score is not None:
            score_percent = quality_score * 100
            if quality_score >= 0.8:
                score_color = "green"
                emoji = "🌟"
            elif quality_score >= 0.6:
                score_color = "yellow"
                emoji = "⭐"
            else:
                score_color = "red"
                emoji = "📊"
            
            self.console.print(
                f"{emoji} [bold]Quality Score:[/bold] [{score_color}]{score_percent:.1f}%[/{score_color}]"
            )
        
        if feedback:
            self.console.print(Panel(
                feedback,
                title="📝 Critic Feedback",
                border_style="yellow",
                padding=(1, 2)
            ))
    
    def show_completion(self, output_path: Optional[str], stats: Dict[str, Any]):
        """Display completion summary.
        
        Args:
            output_path: Path to the generated book
            stats: Statistics about the generated book
        """
        self.console.print()
        self.console.print(Panel.fit(
            "[bold green]✨ Book Composition Completed! ✨[/bold green]",
            border_style="green"
        ))
        
        if output_path:
            self.console.print(f"📖 [bold]Output File:[/bold] [cyan]{output_path}[/cyan]")
        
        if stats:
            self.console.print()
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in stats.items():
                table.add_row(f"  {key}:", str(value))
            
            self.console.print(table)
    
    @contextmanager
    def agent_context(self, agent_name: str, phase_description: str):
        """Context manager for agent execution.
        
        Args:
            agent_name: Name of the agent
            phase_description: Description of what the agent does
            
        Yields:
            Self for method chaining
        """
        style_map = {
            "Planner": ("bold blue", "🎯"),
            "Executor": ("bold green", "⚙️"),
            "Critic": ("bold yellow", "🔍")
        }
        
        style, emoji = style_map.get(agent_name, ("bold cyan", "🤖"))
        
        self.console.print()
        self.console.print(Panel(
            f"[{style}]{emoji} {agent_name} Agent[/{style}]\n{phase_description}",
            border_style=style.split()[-1] if " " in style else style,
            expand=False
        ))
        
        try:
            yield self
        finally:
            self.console.print(f"[dim]  ✓ {agent_name} phase complete[/dim]")


# Global progress display instance
progress = ProgressDisplay()


def show_workflow_start(input_dir: str, output_dir: str, config: Dict[str, Any]):
    """Display workflow startup information.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        config: Configuration dictionary
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🚀 AI Book Composer - Deep-Agent Workflow[/bold cyan]\n"
        "Generating comprehensive books using AI",
        border_style="cyan"
    ))
    
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("📂 Input Directory", input_dir)
    table.add_row("📁 Output Directory", output_dir)
    table.add_row("📖 Book Title", config.get("book_title", "N/A"))
    table.add_row("✍️  Author", config.get("book_author", "N/A"))
    table.add_row("🌍 Language", config.get("language", "N/A"))
    
    console.print(table)
    console.print()


def show_node_transition(from_node: Optional[str], to_node: str, reason: str = ""):
    """Display transition between workflow nodes.
    
    Args:
        from_node: Previous node name (None if starting)
        to_node: Next node name
        reason: Reason for transition
    """
    if from_node:
        arrow = "→"
        console.print(
            f"[dim]  {from_node}[/dim] {arrow} [bold cyan]{to_node}[/bold cyan]"
            + (f" [dim]({reason})[/dim]" if reason else "")
        )
    else:
        console.print(f"[bold cyan]▶ Starting: {to_node}[/bold cyan]")
