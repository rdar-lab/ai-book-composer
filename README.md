# AI Book Composer

Using AI and Deep Agent pattern to generate comprehensive books from source files.

## Overview

AI Book Composer is a tool that automatically generates high-quality books from a directory of source files using AI and the Deep-Agent architecture pattern. It supports multiple input formats (text, audio, video) and generates well-structured books with proper formatting, table of contents, and references.

## Features

- **Multi-format Support**: Process text files, audio files (with transcription), and video files (with transcription)
- **Image Support**: Extract images from PDF files and embed them in generated books
- **Deep-Agent Architecture**: Implements Plan → Execute → Decorate → Iterate → Verify workflow
- **Multiple LLM Providers**: Supports OpenAI GPT, Google Gemini, Azure OpenAI, Ollama (server), and Embedded Ollama (in-process)
- **No API Keys Required**: Default configuration uses embedded ollama and local whisper - runs completely offline
- **LangGraph Orchestration**: Uses LangGraph for robust workflow management
- **Quality-Focused**: Iterative refinement with critic feedback for high-quality output
- **Comprehensive Output**: Generates books with title page, table of contents, chapters, references, and embedded images in RTF format

## Architecture

The system follows the Deep-Agent pattern with four phases:

### Phase 1: The Pr- Processor
- File listing and reading
- Read all text files
- Extract images from PDF files
- Transcribe audio files
- Describe images

### Phase 2: The Planner (Product Manager)
- Analyzes input files
- Creates a structured plan for book generation
- Determines chapter structure and content mapping

### Phase 3: The Executor (Worker)
- Executes tasks using specialized tools:
  - Chapter list generation
  - Chapter generation
  - Book compilation
- Generates content based on the plan

### Phase 4: The Decorator (Visual Content Specialist)
- Analyzes chapter content and available images
- Decides optimal image placement in chapters
- Ensures images enhance reader understanding
- Limits images per chapter to maintain readability

### Phase 5: The Critic (Quality Assurance)
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
  provider: ollama_embedded  # Options: openai, gemini, azure, ollama, ollama_embedded
  model: llama-3.2-3b-instruct

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
  ollama_embedded:
    # Embedded (in-process) ollama execution using llama.cpp
    internal:
        n_ctx: 2048  # Context window size
        n_threads: 4  # Number of CPU threads to use
        n_batch: 64
    run_on_gpu: false  # Use GPU acceleration if available

# Whisper Configuration (for audio/video transcription)
whisper:
  mode: local  # Options: local, remote (default: local for no API keys needed)

# Text file reading configuration
text_reading:
  max_lines_per_read: 100  # Max lines to read at once from text files

# Book generation configuration
book:
  output_language: en-US
  default_title: Composed Book
  default_author: AI Book Composer
  quality_threshold: 0.7
  max_iterations: 3
  style_instructions: ""  # Optional: Guide the AI on book style

# Parallel execution configuration
parallel:
  parallel_execution: 1  # 1 = enabled, 0 = disabled (default: enabled)
  parallel_workers: 4    # Number of parallel workers (default: 4)
```

### Book Style Instructions

You can guide the AI on what kind of book to generate by providing style instructions. This helps the AI tailor the tone, language, and structure to your preferences.

Examples of style instructions:
- `"I want an academic book"` - Produces formal, scholarly content with technical precision
- `"I want it to be light reading"` - Creates accessible, easy-to-read content
- `"I want it to be professional reading material"` - Generates business-appropriate, polished content
- `"I want it to be kids/fun reading material"` - Makes content engaging and appropriate for younger audiences

To use style instructions, add them to your configuration file:

```yaml
book:
  style_instructions: "I want an academic book with formal language and in-depth analysis"
```

Or provide them via the command line:

```bash
ai-book-composer -i input -o output --style-instructions "I want it to be light reading"
```

**Note**: Style instructions are optional. If not provided, the AI will generate content in a neutral, informative style.

### Parallel Execution

The AI Book Composer supports parallel execution to significantly speed up processing of large document collections. When enabled, the system can:

- **Transcribe multiple audio/video files concurrently** - Process media files in parallel using multiple workers
- **Extract images from multiple PDFs simultaneously** - Speed up image extraction from large PDF collections
- **Generate multiple chapters in parallel** - Dramatically reduce book generation time by creating chapters concurrently

To configure parallel execution:

```yaml
parallel:
  parallel_execution: true  # Set to false to disable parallel execution
  parallel_workers: 4       # Number of worker threads (adjust based on your CPU cores)
```

**Performance Impact**: With a 400-file directory, parallel execution can reduce processing time from ~4 hours to significantly less, depending on your hardware and the number of workers configured.

**Note**: For optimal performance, set `parallel_workers` to match the number of CPU cores available on your system.

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
  --max-iterations 3 \
  --style-instructions "I want an academic book"
```

### Options

- `--input-dir, -i`: Directory containing source files (required)
- `--output-dir, -o`: Directory for output files (required)
- `--title, -t`: Book title (default: "Composed Book")
- `--author, -a`: Book author (default: "AI Book Composer")
- `--language, -l`: Target language (default: "en-US")
- `--max-iterations`: Maximum revision iterations (default: 3)
- `--style-instructions, -s`: Instructions to guide the AI on book style (optional)

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
    max_iterations=3,
    style_instructions="I want an academic book"  # Optional
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
- `he` - Hebrew
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
   - Describe the images using a specialized vision LLM

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

### Ollama (Server-based)
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

### Embedded Ollama (Default - No Server Required)

The default configuration uses embedded ollama execution, which runs models in-process without requiring an external server or API keys. Models are automatically downloaded from Hugging Face on first use.

```bash
# Just run - models download automatically!
ai-book-composer -i input -o output
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

### Example 4: Academic Book with Style Instructions

```bash
# Generate an academic-style book from research papers
ai-book-composer \
  -i ./research-papers \
  -o ./output \
  -t "Research Compilation" \
  -a "Dr. Jane Smith" \
  --style-instructions "I want an academic book with formal language, in-depth analysis, and proper citation style"
```

### Example 5: Light Reading Book

```bash
# Generate a casual, easy-to-read book
ai-book-composer \
  -i ./blog-posts \
  -o ./output \
  -t "Easy Reading Collection" \
  -a "John Doe" \
  --style-instructions "I want it to be light reading with a conversational tone and simple explanations"
```

### Example 6: Professional Book

```bash
# Generate a professional business book
ai-book-composer \
  -i ./business-documents \
  -o ./output \
  -t "Business Insights" \
  -a "Corporate Authors" \
  --style-instructions "I want it to be professional reading material suitable for executives and managers"
```
