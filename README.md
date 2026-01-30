# AI Book Composer

Using AI and Deep Agent pattern to generate comprehensive books from source files.

## Overview

AI Book Composer is a tool that automatically generates high-quality books from a directory of source files using AI and the Deep-Agent architecture pattern. It supports multiple input formats (text, audio, video) and generates well-structured books with proper formatting, table of contents, and references.

## Features

- **Multi-format Support**: Process text files, audio files (with transcription), and video files (with transcription)
- **Image Support**: Extract images from PDF files and embed them in generated books
- **Deep-Agent Architecture**: Implements Plan → Execute → Decorate → Iterate → Verify workflow
- **Multiple LLM Providers**: Supports OpenAI GPT, Google Gemini, Azure OpenAI, and Ollama
- **LangGraph Orchestration**: Uses LangGraph for robust workflow management
- **Quality-Focused**: Iterative refinement with critic feedback for high-quality output
- **Comprehensive Output**: Generates books with title page, table of contents, chapters, references, and embedded images in RTF format

## Architecture

The system follows the Deep-Agent pattern with four phases:

### Phase 1: The Planner (Product Manager)
- Analyzes input files
- Creates a structured plan for book generation
- Determines chapter structure and content mapping

### Phase 2: The Executor (Worker)
- Executes tasks using specialized tools:
  - File listing and reading
  - Audio/video transcription (ffmpeg + faster-whisper)
  - Image extraction from PDFs
  - Image listing from input directory
  - Chapter generation
  - Book compilation
- Generates content based on the plan

### Phase 3: The Decorator (Visual Content Specialist)
- Analyzes chapter content and available images
- Decides optimal image placement in chapters
- Ensures images enhance reader understanding
- Limits images per chapter to maintain readability

### Phase 4: The Critic (Quality Assurance)
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

The project uses YAML configuration files. Copy `config.yaml` and customize as needed:

```yaml
# LLM Configuration
llm:
  provider: openai  # Options: openai, gemini, azure, ollama
  model: gpt-4

# Provider-specific settings (use environment variables for API keys)
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  gemini:
    api_key: ${GOOGLE_API_KEY}
  azure:
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    deployment: ${AZURE_OPENAI_DEPLOYMENT}
  ollama:
    base_url: http://localhost:11434
    model: llama2  # Specify Ollama model name

# Text file reading configuration
text_reading:
  max_lines_per_read: 100  # Max lines to read at once from text files

# Book generation configuration
book:
  output_language: en-US
  quality_threshold: 0.7
  max_iterations: 3
```

You can specify a custom config file using the `--config` parameter:

```bash
ai-book-composer --config my-config.yaml -i input -o output
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
- `.ogg` - OGG audio

### Video Files
- `.mp4` - MP4 video
- `.avi` - AVI video
- `.mov` - MOV video
- `.mkv` - MKV video

## Audio and Video Transcription

The system uses OpenAI's Whisper model via the `faster-whisper` library for transcribing audio and video files. The transcription feature includes:

### Multi-Language Support
- **Automatic Language Detection**: The system automatically detects the language of audio/video content
- **Hebrew Support**: Fully tested and validated for Hebrew language transcription (language code: `he`)
- **100+ Languages**: Supports all languages recognized by Whisper including English, Spanish, French, German, Arabic, Chinese, Japanese, and many more

### Caching Mechanism
To improve performance and avoid re-transcribing the same files:
- **Automatic Caching**: Transcription results are automatically cached in hidden files
- **Cache Format**: `.{original-filename}[_language].txt` (e.g., `.audio.mp3.txt` or `.audio.mp3_he.txt` for Hebrew)
- **Language-Aware**: Different language transcriptions are cached separately
- **Cache Location**: Cache files are stored in the same directory as the source audio/video file
- **Cache Reuse**: If a cached transcription exists for the same file and language, it will be used instead of re-transcribing
- **Example**: 
  - Transcribing `interview.mp3` (auto-detect) creates `.interview.mp3.txt`
  - Transcribing `interview.mp3` (Hebrew) creates `.interview.mp3_he.txt`

### Configuration

Configure transcription in `config.yaml`:

```yaml
# Whisper Configuration (for audio/video transcription)
whisper:
  mode: local  # Options: local, remote
  model_size: base  # Options: tiny, base, small, medium, large
  language: null  # Optional: Force specific language (e.g., 'en', 'he', 'es'). If null, auto-detects.
  # For local mode:
  local:
    device: cpu  # Options: cpu, cuda
    compute_type: int8

# Audio/Video Processing Configuration
media_processing:
  chunk_duration: 300  # Chunk size for large files (in seconds)
  max_file_duration: 3600  # Max 1 hour per file
```

### Language Specification

You can specify the language when transcribing to improve accuracy:

```python
# Auto-detect language
result = audio_transcriber.run("file.mp3")

# Force Hebrew transcription
result = audio_transcriber.run("hebrew_audio.mp3", language="he")

# Force English transcription
result = audio_transcriber.run("english_audio.mp3", language="en")
```

Common language codes:
- `en` - English
- `he` - Hebrew (עברית)
- `es` - Spanish
- `fr` - French
- `de` - German
- `ar` - Arabic
- `zh` - Chinese
- `ja` - Japanese
- `ru` - Russian

## Image Processing

The system can extract images from PDF files and embed them into the generated book:

### Features
- **Automatic Image Extraction**: Extracts images from PDF files during content gathering
- **Existing Image Support**: Recognizes and uses existing image files in the input directory
- **AI-Powered Placement**: The Decorator agent uses AI to determine optimal image placement in chapters
- **Format Support**: Supports JPG, JPEG, PNG, GIF, and BMP image formats
- **Smart Positioning**: Places images at start, middle, or end of chapters based on content relevance

### Configuration

Configure image processing in `config.yaml`:

```yaml
# Image Processing Configuration
image_processing:
  supported_formats:
    - jpg
    - jpeg
    - png
    - gif
    - bmp
  extract_from_pdf: true  # Extract images from PDF files
  max_image_size_mb: 10  # Maximum size per image
  max_images_per_chapter: 5  # Maximum images to place per chapter
```

### How It Works

1. **Image Gathering**: During the content gathering phase, the system:
   - Lists all existing image files in the input directory
   - Extracts images from PDF files
   - Stores image metadata (path, format, source)

2. **Image Decoration**: After chapters are generated, the Decorator agent:
   - Analyzes each chapter's content
   - Reviews available images
   - Decides which images are relevant to each chapter
   - Determines optimal placement positions (start, middle, or end)
   - Provides reasoning for each placement decision

3. **Book Generation**: The final RTF book embeds images:
   - Images are inserted at the determined positions
   - Captions are added with the reasoning for each image
   - Images are properly formatted for RTF output

## Output Format

The tool generates an RTF (Rich Text Format) book with:

1. **Title Page**: Book title, author, and date
2. **Table of Contents**: List of all chapters
3. **Chapters**: Generated content organized by chapters with embedded images
4. **References**: List of source files used

## Tools

The system uses the following specialized tools:

- **FileListingTool**: Lists all files in input directory
- **ImageListingTool**: Lists all image files in input directory
- **ImageExtractorTool**: Extracts images from PDF files
- **TextFileReaderTool**: Reads text files with line range support
- **AudioTranscriptionTool**: Transcribes audio using faster-whisper
- **VideoTranscriptionTool**: Extracts and transcribes video audio
- **ChapterWriterTool**: Writes individual chapters
- **ChapterListWriterTool**: Saves chapter planning
- **BookGeneratorTool**: Generates final RTF book

### Tool Integration with LangChain

Tools are exposed to the LLM through LangChain's tool binding system, making them accessible during the execution flow. The `ToolRegistry` class manages all tools and converts them to LangChain-compatible format:

```python
from ai_book_composer.langchain_tools import ToolRegistry

# Initialize tools
registry = ToolRegistry(input_directory, output_directory)

# Get LangChain tools for binding to LLM
tools = registry.get_langchain_tools()

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
```

The executor agent automatically binds these tools to its LLM instance, allowing the AI to discover and use them during task execution. This architecture makes it easy to add new tools without modifying the core workflow.

### MCP Server (Optional)

For external integrations, tools can also be exposed through the **Model Context Protocol (MCP)** as a standalone server. This is useful for testing tools independently or integrating with other MCP-compatible clients:

```bash
# Run standalone MCP server (optional)
python run_mcp_server.py /path/to/input /path/to/output
```

The MCP server provides the same tools through a standardized protocol but is **not required** for normal operation. The main workflow uses tools directly through LangChain's embedded tool system.

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
