"""Integration test using Docker and Ollama."""

import pytest
import subprocess
import time
import os
from pathlib import Path
import yaml

# Skip if running in CI without Docker
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") and not os.environ.get("DOCKER_AVAILABLE"),
    reason="Docker not available in CI"
)


@pytest.fixture(scope="module")
def ollama_container():
    """Start Ollama container with tinyllama model."""
    container_name = "test-ollama-ai-book-composer"
    
    print("Starting Ollama container...")
    # Start Ollama container
    subprocess.run([
        "docker", "run", "-d",
        "--name", container_name,
        "-p", "11434:11434",
        "ollama/ollama"
    ], check=True)
    
    # Wait for container to be ready
    time.sleep(5)
    
    # Pull tinyllama model
    print("Pulling tinyllama model...")
    subprocess.run([
        "docker", "exec", container_name,
        "ollama", "pull", "tinyllama"
    ], check=True)
    
    # Wait for model to be ready
    time.sleep(5)
    
    yield f"http://localhost:11434"
    
    # Cleanup
    print("Stopping Ollama container...")
    subprocess.run(["docker", "stop", container_name], check=False)
    subprocess.run(["docker", "rm", container_name], check=False)


@pytest.fixture
def test_config(tmp_path, ollama_container):
    """Create test configuration."""
    config = {
        'llm': {
            'provider': 'ollama',
            'model': 'tinyllama',
            'temperature': {
                'planning': 0.3,
                'execution': 0.7,
                'critique': 0.2
            }
        },
        'providers': {
            'ollama': {
                'base_url': ollama_container,
                'model': 'tinyllama'
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
    from ai_book_composer import BookComposerWorkflow
    from ai_book_composer.config import Settings
    
    # Load test configuration
    settings = Settings(test_config)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    print(f"Running book generation test...")
    print(f"Input: {test_input}")
    print(f"Output: {output_dir}")
    
    # Create workflow
    workflow = BookComposerWorkflow(
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


def test_ollama_connection(ollama_container):
    """Test connection to Ollama container."""
    import requests
    
    response = requests.get(f"{ollama_container}/api/tags")
    assert response.status_code == 200
    
    tags = response.json()
    model_names = [model["name"] for model in tags.get("models", [])]
    assert "tinyllama:latest" in model_names or "tinyllama" in str(tags)
    
    print(f"✓ Ollama connection test passed!")
    print(f"  - Available models: {model_names}")
