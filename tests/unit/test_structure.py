"""Basic structure tests for AI Book Composer."""

from src.ai_book_composer import config
from src.ai_book_composer.agents import state
from src.ai_book_composer.agents.state import create_initial_state
from src.ai_book_composer.config import Settings


def test_module_structure():
    """Test that all modules are properly structured."""

    print("Testing module structure...")

    assert hasattr(config, 'Settings')
    print("✓ Config module OK")

    assert hasattr(state, 'AgentState')
    assert hasattr(state, 'create_initial_state')
    print("✓ Agents state module OK")


def test_state_creation():
    """Test state creation."""

    print("\nTesting state creation...")

    state_instance = create_initial_state(
        input_directory="/tmp/input",
        output_directory="/tmp/output",
        language="en-US",
        book_title="Test Book",
        book_author="Test Author"
    )

    assert state_instance['input_directory'] == "/tmp/input"
    assert state_instance['output_directory'] == "/tmp/output"
    assert state_instance['language'] == "en-US"
    assert state_instance['book_title'] == "Test Book"
    assert state_instance['book_author'] == "Test Author"
    assert state_instance['status'] == "initialized"
    assert state_instance['iterations'] == 0
    assert isinstance(state_instance['files'], list)
    assert isinstance(state_instance['chapters'], list)


def test_config():
    """Test configuration."""

    print("\nTesting configuration...")

    settings = Settings()
    llm_config = settings.llm
    assert hasattr(llm_config, 'provider')
    assert hasattr(llm_config, 'model')
    book_config = settings.book
    assert hasattr(book_config, 'output_language')

    # Test default values
    assert book_config.output_language == "en-US"
