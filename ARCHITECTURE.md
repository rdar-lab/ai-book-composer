# AI Book Composer - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI BOOK COMPOSER                            │
│                  Deep-Agent Architecture                        │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   User Input     │
                    │  - Source Files  │
                    │  - Book Title    │
                    │  - Language      │
                    │  - Style Guide   │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   LangGraph Workflow     │
              │   State Management       │
              └──────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │         PHASE 1: PRE-PROCESSOR         │
        │     "Content Gathering Specialist"     │
        ├────────────────────────────────────────┤
        │ • List all input files                 │
        │ • Read text files (.txt, .md, .pdf)    │
        │ • Transcribe audio (.mp3, .wav, etc.)  │
        │ • Transcribe video audio tracks        │
        │   (.mp4, .mov, etc.)                   │
        │ • Extract images from PDFs             │
        │ • Describe images with vision AI       │
        │ • Initialize RAG vector database       │
        │ • Extract key terms from content       │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │         PHASE 2: PLANNER               │
        │         "Product Manager"              │
        ├────────────────────────────────────────┤
        │ • Analyze gathered content             │
        │ • Create structured task plan          │
        │ • Define chapter structure             │
        │ • Map content to chapters              │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │         PHASE 3: EXECUTOR              │
        │            "Worker"                    │
        ├────────────────────────────────────────┤
        │ Tasks (executed sequentially):         │
        │ • Plan Chapters - Create chapter list  │
        │ • Generate Chapters - Write content    │
        │   (uses RAG for content retrieval)     │
        │ • Compile References - Source list     │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │         PHASE 4: DECORATOR             │
        │    "Visual Content Specialist"         │
        ├────────────────────────────────────────┤
        │ • Analyze chapter content              │
        │ • Match images to chapters             │
        │ • AI-powered placement decisions       │
        │ • Position images (start/middle/end)   │
        │ • Ensure readability (max per chapter) │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │         PHASE 5: CRITIC                │
        │         "QA Review"                    │
        ├────────────────────────────────────────┤
        │ • Evaluate overall book quality        │
        │ • Provide constructive feedback        │
        │ • Calculate quality score              │
        │ • Decision: Approve or Revise          │
        └────────────────┬───────────────────────┘
                         │
               ┌─────────┴─────────┐
               │                   │
         ▼ (Approved)        ▼ (Revise & Iterate)
    ┌──────────┐            └──────────┐
    │  WRITER  │                       │
    │  PHASE   │             ┌─────────▼────────┐
    └────┬─────┘             │ Max Iterations?  │
         │                   └─────────┬────────┘
         │                       No ◀──┘   │ Yes
         │                        │        │
         │                        │        ▼
         │                Back to ▼     Force
         │               Executor      Approve
         │                   │            │
         └───────────────────┴────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Final Book    │
                    │ ─────────────  │
                    │ • Title Page   │
                    │ • TOC          │
                    │ • Chapters     │
                    │ • Images       │
                    │ • References   │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  DOCX Output   │
                    └────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LLM PROVIDERS                              │
├─────────────────────────────────────────────────────────────────┤
│ OpenAI GPT │ Gemini │ Azure OpenAI │ Ollama │ Embedded Ollama  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SUPPORTED FILE TYPES                         │
├─────────────────────────────────────────────────────────────────┤
│  Text: .txt, .md, .rst, .docx, .rtf, .pdf                       │
│  Audio: .mp3, .wav, .m4a, .flac, .ogg                           │
│  Video: .mp4, .avi, .mov, .mkv                                  │
│  Images: .jpg, .jpeg, .png, .gif, .bmp (extracted from PDFs)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    KEY TECHNOLOGIES                             │
├─────────────────────────────────────────────────────────────────┤
│ • LangGraph - Workflow orchestration and state management       │
│ • RAG/Vector DB - Content retrieval for chapter generation      │
│ • Whisper - Audio/video transcription (local or API)            │
│ • Vision AI - Image description and analysis                    │
│ • Parallel Processing - Concurrent file processing              │
└─────────────────────────────────────────────────────────────────┘
```

## Workflow Sequence

1. **Initialization**: User provides input directory, output settings, and book metadata
2. **PreProcessor Phase**: 
   - Scans input directory for all supported files
   - Reads text files (.txt, .md, .rst, .pdf, .docx, .rtf)
   - Transcribes audio files using Whisper (.mp3, .wav, .m4a, .flac, .ogg)
   - Transcribes video files using Whisper (extracts and transcribes audio track from .mp4, .avi, .mov, .mkv)
   - Extracts images from PDF files
   - Describes extracted images using vision AI
   - Initializes RAG vector database with all content
   - Extracts key terms for better context understanding
3. **Planning Phase**: 
   - Planner agent analyzes all gathered content
   - Creates structured execution plan with tasks:
     * Plan Chapters
     * Generate Chapters  
     * Compile References
4. **Execution Phase**: Executor runs tasks sequentially:
   - **Plan Chapters**: Creates comprehensive chapter list structure
   - **Generate Chapters**: Writes chapter content using RAG for content retrieval
   - **Compile References**: Builds list of all source files used
5. **Decoration Phase**: 
   - Decorator agent analyzes chapter content and available images
   - Makes AI-powered decisions on image placement
   - Positions images at optimal locations (start, middle, or end of chapters)
   - Limits images per chapter for readability
6. **Critique Phase**: 
   - Critic evaluates overall book quality
   - Provides constructive feedback
   - Calculates quality score
   - Decides to approve or request revisions
7. **Iteration**: If quality is below threshold and max iterations not reached:
   - Reset task index and clear caches
   - Return to Execution phase for revision
8. **Writer Phase**: Once approved or max iterations reached:
   - Generates final DOCX book with all chapters, images, and formatting
9. **Finalization**: Complete workflow and output final book file

## Quality Control

The system implements a comprehensive quality control approach:

1. **Caching for Performance**: Results are cached to improve performance
   - Chapter list structure
   - Chapter content
   - Image decorations
   - Caches are cleared when iterations require revisions

2. **Final Critic Phase**: Evaluates complete book quality
   - Reviews all chapters holistically
   - Analyzes content quality, coherence, and completeness
   - Provides detailed constructive feedback for improvements
   - Calculates quality score (0-1 scale)
   - Triggers iteration if quality is below threshold

3. **Iterative Refinement Loop**:
   - If quality score < threshold: Reset and revise
   - If quality score >= threshold: Approve and proceed to Writer
   - Maximum iterations enforced to prevent infinite loops
   - After max iterations, book is force-approved

## Key Features

- **Deep-Agent Architecture**: 5-phase workflow (PreProcessor → Planner → Executor → Decorator → Critic)
- **Iterative Refinement**: Multiple revision cycles for quality improvement
- **Multi-Format Input**: Seamlessly handles text, audio, video, and images
- **Image Support**: Extracts images from PDFs, describes them with vision AI, and intelligently places them in chapters
- **RAG-Enhanced Content**: Uses vector database for intelligent content retrieval during chapter generation
- **Parallel Processing**: Concurrent file processing for faster execution
- **Flexible LLM Backend**: Works with OpenAI, Gemini, Azure OpenAI, Ollama server, or Embedded Ollama
- **Local Execution**: Can run completely offline with embedded Ollama and local Whisper
- **Rich Output**: Professional DOCX book format with title page, TOC, chapters, images, and references
- **State Management**: LangGraph handles complex workflow states and transitions
- **Quality Focus**: Prioritizes output quality over speed with comprehensive critique phase
