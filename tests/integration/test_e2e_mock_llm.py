import importlib
import json
import shutil
from pathlib import Path

from src.ai_book_composer import logging_config
from src.ai_book_composer.config import Settings
from src.ai_book_composer.workflow import BookComposerWorkflow


class FakeLLM:
    """Minimal fake LLM that returns shaped responses based on prompt content.

    The real system calls `get_llm(settings, ...)` which returns a BaseChatModel-like
    object with `.invoke()` and `.bind_tools(...)`. Agents call `.invoke()` with
    a string containing both system and user prompts. We inspect that string to
    return stage-appropriate short outputs so the workflow can proceed.
    """

    def __init__(self):
        self.tools_bound = False

    def bind_tools(self, tools, **kwargs):
        # Returning self is enough; callers expect an object with invoke(...)
        self.tools_bound = True
        return self

    def invoke(self, prompt_or_messages):
        print(f'FAKE LLM - Got prompt/messages: {prompt_or_messages}')
        # Normalize to string
        content = prompt_or_messages
        try:
            content = content if isinstance(content, str) else json.dumps(content)
        except Exception:
            content = str(prompt_or_messages)

        lower = content.lower()

        # Summarizer: return a short summary string
        if "summarize the following source file" in lower:
            response = "This is a file summary."

        # Images descriptor: return text
        elif "description of the image" in lower:
            response = "A futuristic cityscape with flying cars and neon lights."

        # Planner: return a JSON plan inside a markdown codeblock or plain JSON
        elif 'detailed step-by-step plan' in lower:
            plan = [
                {"task": "write_chapter", "title": "Introduction", "description": "Intro chapter"},
                {"task": "write_chapter", "title": "Background", "description": "Background chapter"},
                {"task": "write_chapter", "title": "Data", "description": "Data chapter"},
                {"task": "write_chapter", "title": "Summary", "description": "Summary chapter"}
            ]
            response =  json.dumps(plan)

        # Chapter Structurer: return a JSON list of chapter metadata (number, title, description)
        elif 'chapter structure' in lower:
            chapters = [
                {
                    "number": 1,
                    "title": "Chapter Title",
                    "description": "Brief description"
                },
                {
                    "number": 2,
                    "title": "Chapter Title",
                    "description": "Brief description"
                },
                {
                    "number": 3,
                    "title": "Chapter Title",
                    "description": "Brief description"
                },
                {
                    "number": 4,
                    "title": "Chapter Title",
                    "description": "Brief description"
                }
            ]
            response =  json.dumps(chapters)

        # Executor: return chapter text. Ensure it's distinguishable
        elif 'write chapter' in lower:
            # Return a simple chapter text
            response = "<result>Chapter Title: Sample\n\nThis is generated chapter content about AI.</result>"

        # Decorator: JSON of the decoration locations
        elif 'visual content specialist' in lower:
            response = "[]"

        # Critic: return a short quality approval
        elif 'professional book critic' in lower:
            response = json.dumps({"quality_score": 0.95, "approval": True})

        else:
            # Default fallback
            response = "OK"

        print(f'-> -> -> FAKE LLM - Returning response: {response}')
        return response


def _copy_fixtures(fixtures_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for file in fixtures_dir.iterdir():
        if file.is_file():
            shutil.copy(file, dest_dir / file.name)

def _patch_get_llm():
    modules = [
        'src.ai_book_composer.llm',
        'src.ai_book_composer.agents.agent_base',
        'src.ai_book_composer.utils.file_utils',
    ]

    for module in modules:
        target__module = importlib.import_module(module)
        if hasattr(target__module, 'get_llm'):
            print('Patching get_llm in', module)
            setattr(target__module, 'get_llm', lambda *a, **k: FakeLLM())

def _patch_agent_base():
    agent_base_module = importlib.import_module('src.ai_book_composer.agents.agent_base')
    agent_base_class = getattr(agent_base_module, 'AgentBase', None)
    if agent_base_class:
        if hasattr(agent_base_class, '_invoke_llm'):
            print('Patching _invoke_llm in', agent_base_class)
            setattr(agent_base_class, '_invoke_llm', lambda *a, **k: FakeLLM().invoke(a[1] + a[2]))
        if hasattr(agent_base_class, '_invoke_agent'):
            print('Patching _invoke_agent in', agent_base_class)
            setattr(agent_base_class, '_invoke_agent', lambda *a, **k: FakeLLM().invoke(a[1] + a[2]))


def test_e2e_workflow_with_only_llm_mocked(tmp_path):
    """Run an end-to-end workflow while mocking only the LLM factory.

    We patch the `get_llm` function on the package's `llm` module before importing
    the `workflow` (and, transitively, the agents and utils). This prevents
    modules from binding the real factory at import time.
    """

    # Arrange: prepare input/output dirs using existing fixtures
    fixtures_dir = Path(__file__).parent.parent / 'fixtures'
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    cache_dir = tmp_path / '.cache'

    _copy_fixtures(fixtures_dir, input_dir)

    _patch_get_llm()
    _patch_agent_base()

    # Create minimal config/settings now that modules are imported
    settings = Settings()
    # Keep logs quiet for test
    settings.logging.level = 'INFO'
    settings.logging.file = None
    settings.logging.console_output = True

    settings.general.cache_dir = str(cache_dir)
    # Make workflow fast
    settings.book.max_iterations = 1
    settings.parallel.parallel_execution = False

    settings.llm.provider = 'fake'
    settings.llm.model = 'fake'

    logging_config.setup_logging(settings)

    # Act: run the workflow
    workflow = BookComposerWorkflow(
        settings=settings,
        input_directory=str(input_dir),
        output_directory=str(output_dir),
        language="en-US",
        book_title="Test Book",
        book_author="Tester",
        max_iterations=1,
    )

    final_state = workflow.run()

    # Assert: basic invariants about final state and output
    assert final_state is not None
    assert final_state.get('status') in ['completed', 'approved']
    assert len(final_state.get('chapters', [])) >= 0

    final_output = final_state.get('final_output_path')
    assert final_output is not None

    out_path = Path(final_output)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Some additional checks: quality score and chapters
    assert final_state.get('quality_score', 0) >= 0
    # At least ensure the output file contains some generated text
    text = out_path.read_text(errors='ignore')
    assert 'AI' in text or 'Chapter' in text or len(text) > 100
