# AI Book Composer

Using AI and Deep Agent pattern to generate comprehensive books from source files.

## Overview

AI Book Composer is a tool that automatically generates high-quality books from a directory of source files using AI and the Deep-Agent architecture pattern. It supports multiple input formats (text, audio, video) and generates well-structured books with proper formatting, table of contents, and references.

## Features

- **Multi-format Support**: Process text files, audio files (with transcription), and video files (with transcription)
- **Deep-Agent Architecture**: Implements Plan → Execute → Iterate → Verify workflow
- **Multiple LLM Providers**: Supports OpenAI GPT, Google Gemini, Azure OpenAI, and Ollama
- **LangGraph Orchestration**: Uses LangGraph for robust workflow management
- **Quality-Focused**: Iterative refinement with critic feedback for high-quality output
- **Comprehensive Output**: Generates books with title page, table of contents, chapters, and references in RTF format

## Architecture

The system follows the Deep-Agent pattern with three phases:

### Phase 1: The Planner (Product Manager)
- Analyzes input files
- Creates a structured plan for book generation
- Determines chapter structure and content mapping

### Phase 2: The Executor (Worker)
- Executes tasks using specialized tools:
  - File listing and reading
  - Audio/video transcription (ffmpeg + faster-whisper)
  - Chapter generation
  - Book compilation
- Generates content based on the plan

### Phase 3: The Critic (Quality Assurance)
- Evaluates generated content quality
- Provides constructive feedback
- Decides whether to approve or request revisions
- Iterates until quality threshold is met

## Installation

```bash
# Clone the repository
git clone https://github.com/rdar-lab/ai-book-composer.git
cd ai-book-composer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Additional Requirements

For audio/video transcription, you need ffmpeg:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Configuration

Create a `.env` file in the project root (use `.env.example` as template):

```env
# LLM Provider (openai, gemini, azure, ollama)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Google Gemini
GOOGLE_API_KEY=your-google-api-key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint
AZURE_OPENAI_DEPLOYMENT=your-deployment

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Book Configuration
OUTPUT_LANGUAGE=en-US
MAX_LINES_PER_READ=100
```

## Usage

### Command Line Interface

```bash
ai-book-composer \
  --input-dir /path/to/source/files \
  --output-dir /path/to/output \
  --title "My Book Title" \
  --author "Author Name" \
  --language en-US \
  --max-iterations 3
```

### Options

- `--input-dir, -i`: Directory containing source files (required)
- `--output-dir, -o`: Directory for output files (required)
- `--title, -t`: Book title (default: "Composed Book")
- `--author, -a`: Book author (default: "AI Book Composer")
- `--language, -l`: Target language (default: "en-US")
- `--max-iterations`: Maximum revision iterations (default: 3)

### Python API

```python
from ai_book_composer import BookComposerWorkflow

# Create workflow
workflow = BookComposerWorkflow(
    input_directory="/path/to/source/files",
    output_directory="/path/to/output",
    language="en-US",
    book_title="My Book",
    book_author="Author Name",
    max_iterations=3
)

# Run workflow
final_state = workflow.run()

# Access results
print(f"Status: {final_state['status']}")
print(f"Output: {final_state['final_output_path']}")
print(f"Quality Score: {final_state['quality_score']}")
```

## Supported File Types

### Text Files
- `.txt` - Plain text
- `.md` - Markdown
- `.rst` - reStructuredText

### Audio Files
- `.mp3` - MP3 audio
- `.wav` - WAV audio
- `.m4a` - M4A audio
- `.flac` - FLAC audio

### Video Files
- `.mp4` - MP4 video
- `.avi` - AVI video
- `.mov` - MOV video
- `.mkv` - MKV video

## Output Format

The tool generates an RTF (Rich Text Format) book with:

1. **Title Page**: Book title, author, and date
2. **Table of Contents**: List of all chapters
3. **Chapters**: Generated content organized by chapters
4. **References**: List of source files used

## Tools

The system uses the following specialized tools:

- **FileListingTool**: Lists all files in input directory
- **TextFileReaderTool**: Reads text files with line range support
- **AudioTranscriptionTool**: Transcribes audio using faster-whisper
- **VideoTranscriptionTool**: Extracts and transcribes video audio
- **ChapterWriterTool**: Writes individual chapters
- **ChapterListWriterTool**: Saves chapter planning
- **BookGeneratorTool**: Generates final RTF book

## Development

### Project Structure

```
ai-book-composer/
├── src/
│   └── ai_book_composer/
│       ├── agents/          # Deep-Agent components
│       │   ├── planner.py   # Planning agent
│       │   ├── executor.py  # Execution agent
│       │   ├── critic.py    # Critique agent
│       │   └── state.py     # State management
│       ├── tools/           # Tool implementations
│       │   ├── base_tools.py
│       │   └── book_generator.py
│       ├── workflow.py      # LangGraph workflow
│       ├── llm.py          # LLM provider abstraction
│       ├── config.py       # Configuration
│       └── cli.py          # CLI interface
├── requirements.txt
├── setup.py
└── README.md
```

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

## LLM Provider Setup

### OpenAI
```bash
export OPENAI_API_KEY=your-key
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4
```

### Google Gemini
```bash
export GOOGLE_API_KEY=your-key
export LLM_PROVIDER=gemini
export LLM_MODEL=gemini-pro
```

### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=your-endpoint
export AZURE_OPENAI_DEPLOYMENT=your-deployment
export LLM_PROVIDER=azure
```

### Ollama (Local)
```bash
# Start Ollama server
ollama serve

# Pull a model
ollama pull llama2

# Configure
export LLM_PROVIDER=ollama
export LLM_MODEL=llama2
export OLLAMA_BASE_URL=http://localhost:11434
```

## Examples

### Example 1: Generate Book from Blog Posts

```bash
ai-book-composer \
  -i ./blog-posts \
  -o ./output \
  -t "Best of My Blog" \
  -a "John Doe"
```

### Example 2: Generate Book from Transcribed Videos

```bash
ai-book-composer \
  -i ./video-lectures \
  -o ./output \
  -t "Video Lecture Series" \
  -a "Professor Smith" \
  -l en-US
```

### Example 3: Mixed Content Book

```bash
# Directory with .txt, .mp3, .mp4 files
ai-book-composer \
  -i ./mixed-content \
  -o ./output \
  -t "Comprehensive Guide" \
  -a "AI Book Composer"
```

## Troubleshooting

### Issue: ffmpeg not found
**Solution**: Install ffmpeg using your system package manager

### Issue: Out of memory during transcription
**Solution**: Use smaller Whisper model size (tiny or base) or process fewer files

### Issue: LLM API errors
**Solution**: Check API keys, rate limits, and internet connection

### Issue: Poor book quality
**Solution**: Increase max_iterations, use better LLM model (e.g., GPT-4), or provide better source content

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- LangGraph for orchestration framework
- Faster Whisper for audio transcription
- OpenAI, Google, and other LLM providers
- FFmpeg for media processing
