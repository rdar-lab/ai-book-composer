"""Basic structure tests for AI Book Composer."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_module_structure():
    """Test that all modules are properly structured."""

    print("Testing module structure...")

    # noinspection PyUnresolvedReferences
    from ai_book_composer import config
    assert hasattr(config, 'Settings')
    print("✓ Config module OK")

    # noinspection PyUnresolvedReferences
    from ai_book_composer.agents import state
    assert hasattr(state, 'AgentState')
    assert hasattr(state, 'create_initial_state')
    print("✓ Agents state module OK")


def test_state_creation():
    """Test state creation."""

    print("\nTesting state creation...")

    # noinspection PyUnresolvedReferences
    from ai_book_composer.agents.state import create_initial_state

    state = create_initial_state(
        input_directory="/tmp/input",
        output_directory="/tmp/output",
        language="en-US",
        book_title="Test Book",
        book_author="Test Author"
    )

    assert state['input_directory'] == "/tmp/input"
    assert state['output_directory'] == "/tmp/output"
    assert state['language'] == "en-US"
    assert state['book_title'] == "Test Book"
    assert state['book_author'] == "Test Author"
    assert state['status'] == "initialized"
    assert state['iterations'] == 0
    assert isinstance(state['files'], list)
    assert isinstance(state['chapters'], list)


def test_config():
    """Test configuration."""

    print("\nTesting configuration...")

    # noinspection PyUnresolvedReferences
    from ai_book_composer.config import Settings

    settings = Settings()
    llm_config = settings.llm
    assert hasattr(llm_config, 'provider')
    assert hasattr(llm_config, 'model')
    book_config = settings.book
    assert hasattr(book_config, 'output_language')
    text_reading_config = settings.text_reading
    assert hasattr(text_reading_config, 'max_lines_per_read')

    # Test default values
    assert book_config.output_language == "en-US"
    assert text_reading_config.max_lines_per_read == 100
