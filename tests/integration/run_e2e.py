"""Integration test using Docker and Ollama."""
import datetime
import shutil
from pathlib import Path

import yaml

from src.ai_book_composer import logging_config
from src.ai_book_composer.config import Settings
from src.ai_book_composer.workflow import BookComposerWorkflow


def generate_config_file(config_dir, logs_dir, cache_dir):
    """Create test configuration."""
    config = {
        'llm': {
            'provider': 'ollama',
            'model': 'qwen2.5:7b-instruct',
            'temperature': {
                'planning': 0.3,
                'execution': 0.7,
                'critique': 0.2
            }
        },
        'providers': {
            'ollama': {
                    'base_url': 'http://localhost:11434'
                }
        },
        'whisper': {
            'mode': 'local',
            'model_size': 'tiny'
        },
        'text_reading': {
            'max_lines_per_read': 100
        },
        'book': {
            'output_language': 'en-US',
            'quality_threshold': 0.5,
            'max_iterations': 1
        },
        'logging': {
            'level': 'DEBUG',
            'file': str(logs_dir / f'run_{datetime.datetime.now()}.log'),
            'console_output': False
        },
        'parallel': {
            'parallel_execution': False,
            'parallel_workers': 1
        },
        'general': {
            'cache_dir': str(cache_dir)
        },
        'vision_model': {
            'provider': 'ollama',
            'model': 'moondream',
            'temperature': 0.3
        },
    }

    config_file = config_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    return str(config_file)


def generate_input_dir(input_dir):
    """Create test input files."""

    # Copy all files from the /tests/fixtures directory to the input directory
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    for file in fixtures_dir.iterdir():
        if file.is_file():
            dest_file = input_dir / file.name
            dest_file.write_bytes(file.read_bytes())

    return str(input_dir)


def run_book_generation_end_to_end():
    """Test complete book generation workflow."""

    # Get project root path
    project_root = Path(__file__).parent.parent.parent

    run_dir = project_root / "run"

    run_dir.mkdir(parents=True, exist_ok=True)

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = run_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = run_dir / "logs"

    if logs_dir.exists():
        shutil.rmtree(logs_dir)

    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate configuration file
    config = generate_config_file(config_dir, logs_dir, cache_dir)
    # Generate input directory
    generate_input_dir(input_dir)

    settings = Settings(config)
    logging_config.setup_logging(settings)

    print(f"Running book generation test...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Create workflow
    workflow = BookComposerWorkflow(
        settings=settings,
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        language="en-US",
        book_title="Test AI Book",
        book_author="Test Author",
        max_iterations=1
    )

    # Run workflow
    final_state = workflow.run()

    # Verify results
    assert final_state["status"] in ["completed", "approved"]
    assert len(final_state.get("chapters", [])) > 0
    assert final_state.get("final_output_path") is not None

    # Verify book file exists
    book_file = Path(final_state["final_output_path"])
    assert book_file.exists()
    assert book_file.stat().st_size > 0

    print("✓ Book generation test passed!")
    print(f"  - Generated {len(final_state['chapters'])} chapters")
    print(f"  - Book saved to: {final_state['final_output_path']}")
    print(f"  - Quality score: {final_state.get('quality_score', 'N/A')}")


if __name__ == "__main__":
    try:
        run_book_generation_end_to_end()
    except Exception as e:
        print(f"✗ Book generation test failed: {repr(e)}")
        # raise e
