"""Basic structure tests for AI Book Composer."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_module_structure():
    """Test that all modules are properly structured."""

    print("Testing module structure...")

    # Test config module (no external deps)
    try:
        from ai_book_composer import config
        assert hasattr(config, 'Settings')
        assert hasattr(config, 'settings')
        print("✓ Config module OK")
    except Exception as e:
        print(f"✗ Config module failed: {e}")
        return False

    # Test main module
    try:
        import ai_book_composer
        assert hasattr(ai_book_composer, '__version__')
        print("✓ Main module OK")
    except Exception as e:
        print(f"✗ Main module failed: {e}")
        return False

    # Test agents state module (no external deps)
    try:
        from ai_book_composer.agents import state
        assert hasattr(state, 'AgentState')
        assert hasattr(state, 'create_initial_state')
        print("✓ Agents state module OK")
    except Exception as e:
        print(f"✗ Agents state module failed: {e}")
        return False

    print("\n✓ All basic structure tests passed!")
    return True


def test_state_creation():
    """Test state creation."""

    print("\nTesting state creation...")

    try:
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

        print("✓ State creation test passed!")
        return True
    except Exception as e:
        print(f"✗ State creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration."""

    print("\nTesting configuration...")

    try:
        from ai_book_composer.config import Settings

        settings = Settings()
        assert hasattr(settings, 'llm_provider')
        assert hasattr(settings, 'llm_model')
        assert hasattr(settings, 'output_language')
        assert hasattr(settings, 'max_lines_per_read')

        # Test default values
        assert settings.output_language == "en-US"
        assert settings.max_lines_per_read == 100

        print("✓ Configuration test passed!")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all expected files exist."""

    print("\nTesting file structure...")

    base_path = Path(__file__).parent

    expected_files = [
        "src/ai_book_composer/__init__.py",
        "src/ai_book_composer/config.py",
        "src/ai_book_composer/llm.py",
        "src/ai_book_composer/workflow.py",
        "src/ai_book_composer/cli.py",
        "src/ai_book_composer/agents/__init__.py",
        "src/ai_book_composer/agents/state.py",
        "src/ai_book_composer/agents/planner.py",
        "src/ai_book_composer/agents/executor.py",
        "src/ai_book_composer/agents/critic.py",
        "src/ai_book_composer/tools/__init__.py",
        "src/ai_book_composer/tools/base_tools.py",
        "src/ai_book_composer/tools/book_generator.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        ".env.example",
        ".gitignore"
    ]

    missing = []
    for file_path in expected_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing.append(file_path)
            print(f"✗ Missing: {file_path}")

    if missing:
        print(f"\n✗ File structure test failed: {len(missing)} files missing")
        return False

    print("✓ All expected files present!")
    return True


def main():
    """Run all tests."""

    print("=" * 60)
    print("AI Book Composer - Structure Tests")
    print("=" * 60)
    print()

    results = []

    results.append(test_file_structure())
    results.append(test_module_structure())
    results.append(test_config())
    results.append(test_state_creation())

    print()
    print("=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print()
        print("Note: These are basic structure tests.")
        print("Full functionality tests require installing dependencies:")
        print("  pip install -r requirements.txt")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
