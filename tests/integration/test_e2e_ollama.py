"""Integration test using Docker and Ollama."""

from pathlib import Path

import pytest
import yaml
from ai_book_composer.workflow import BookComposerWorkflow
from ai_book_composer.config import Settings
from ai_book_composer import config

@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    config = {
        'llm': {
            'provider': 'ollama_embedded',
            'model': 'tinyllama',
            'temperature': {
                'planning': 0.3,
                'execution': 0.7,
                'critique': 0.2
            }
        },
        'providers': {
            'ollama_embedded': {
                'model': 'tinyllama',
                'n_ctx': 2048,
                'n_threads': 4,
                'run_on_gpu': False,
                'verbose': False
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
            'level': 'INFO',
            'file': str(tmp_path / 'test.log'),
            'console_output': True
        },
        'parallel': {
            'parallel_execution': False,
            'parallel_workers': 1
        }
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    return str(config_file)


@pytest.fixture
def test_input(tmp_path):
    """Create test input files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test text files
    (input_dir / "article1.txt").write_text(
        "Introduction to AI\n\n"
        "Artificial Intelligence is a fascinating field.\n"
        "It encompasses many technologies and approaches.\n"
    )

    (input_dir / "article2.txt").write_text(
        "Machine Learning Basics\n\n"
        "Machine learning is a subset of AI.\n"
        "It focuses on pattern recognition and learning from data.\n"
    )

    return str(input_dir)


def test_book_generation_end_to_end(test_config, test_input, tmp_path):
    """Test complete book generation workflow."""

    settings = Settings(test_config)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    print(f"Running book generation test...")
    print(f"Input: {test_input}")
    print(f"Output: {output_dir}")

    # Create workflow
    workflow = BookComposerWorkflow(
        settings=settings,
        input_directory=test_input,
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
