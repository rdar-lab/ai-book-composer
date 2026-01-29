# Quick Start Guide

## 1. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Configuration

Create `.env` file:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
OUTPUT_LANGUAGE=en-US
```

## 3. Run

```bash
ai-book-composer -i ./source-files -o ./output -t "My Book" -a "Author"
```

## 4. Example

```bash
cd examples
python run_example.py
```

## Options

- `-i, --input-dir`: Source files directory (required)
- `-o, --output-dir`: Output directory (required)
- `-t, --title`: Book title (default: "Composed Book")
- `-a, --author`: Author name (default: "AI Book Composer")
- `-l, --language`: Language code (default: "en-US")
- `--max-iterations`: Max revision cycles (default: 3)

## Supported File Types

**Text**: .txt, .md, .rst
**Audio**: .mp3, .wav, .m4a, .flac
**Video**: .mp4, .avi, .mov, .mkv

## LLM Providers

### OpenAI
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...
```

### Gemini
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-pro
GOOGLE_API_KEY=...
```

### Azure
```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
```

### Ollama (Local)
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

## Output

Book generated in RTF format with:
- Title page
- Table of contents
- Chapters
- References

## Troubleshooting

**ffmpeg not found**: Install ffmpeg for your OS
**API errors**: Check API keys and rate limits
**Out of memory**: Use smaller Whisper model or fewer files
**Poor quality**: Increase max-iterations or use better LLM

## Python API

```python
from ai_book_composer import BookComposerWorkflow

workflow = BookComposerWorkflow(
    input_directory="/path/to/files",
    output_directory="/path/to/output",
    language="en-US",
    book_title="My Book",
    book_author="Author Name",
    max_iterations=3
)

result = workflow.run()
print(f"Book created: {result['final_output_path']}")
```
