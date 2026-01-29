# AI Book Composer - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI BOOK COMPOSER                             │
│                  Deep-Agent Architecture                         │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   User Input     │
                    │  - Source Files  │
                    │  - Book Title    │
                    │  - Language      │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   LangGraph Workflow     │
              │   State Management       │
              └──────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│   PHASE 1      │  │   PHASE 2      │  │   PHASE 3      │
│   PLANNER      │  │   EXECUTOR     │  │   CRITIC       │
│ "Product Mgr"  │  │   "Worker"     │  │ "QA Review"    │
└────────────────┘  └────────────────┘  └────────────────┘
         │                   │                   │
         │ Creates Plan      │ Uses Tools        │ Validates
         ▼                   ▼                   ▼
    ┌─────────┐      ┌──────────────┐    ┌──────────┐
    │ Tasks   │      │   Tools      │    │ Feedback │
    │ - Gather│      │ ────────────  │    │ Quality  │
    │ - Plan  │      │ File Lister  │    │ Score    │
    │ - Write │      │ Text Reader  │    │ Decision │
    │ - Compile│     │ Audio Trans. │    └─────┬────┘
    └────┬────┘      │ Video Trans. │          │
         │           │ Chapter Write│          │
         └──────────▶│ Book Gen.    │◀─────────┘
                     └──────┬───────┘
                            │              ┌──────────┐
                            ├─────────────▶│ Iterate? │
                            │              └─────┬────┘
                            │                    │
                            │         Yes ◀──────┘ No
                            │          │           │
                            │          └───────────┤
                            ▼                      ▼
                     ┌──────────────┐      ┌──────────┐
                     │ Final Book   │      │ Approved │
                     │ - Title Page │      │ Complete │
                     │ - TOC        │      └──────────┘
                     │ - Chapters   │
                     │ - References │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  RTF Output  │
                     └──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LLM PROVIDERS                               │
├─────────────────────────────────────────────────────────────────┤
│  OpenAI GPT  │  Google Gemini  │  Azure OpenAI  │  Ollama      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SUPPORTED FILE TYPES                          │
├─────────────────────────────────────────────────────────────────┤
│  Text: .txt, .md, .rst, .docx, .rtf, .pdf                      │
│  Audio: .mp3, .wav, .m4a, .flac, .ogg                          │
│  Video: .mp4, .avi, .mov, .mkv                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Workflow Sequence

1. **Initialization**: User provides input directory, output settings, and book metadata
2. **File Discovery**: System scans input directory for all supported files
3. **Planning Phase**: Planner analyzes files and creates structured plan
4. **Execution Phase**: Executor runs tasks sequentially:
   - Gathers content (reads text, transcribes audio/video)
   - Plans chapter structure
   - Generates chapter content using LLM
   - Compiles references
   - Generates final RTF book
5. **Critique Phase**: Critic evaluates quality and provides feedback
6. **Iteration**: If quality is below threshold, return to execution
7. **Finalization**: Once approved, output final book

## Key Features

- **Iterative Refinement**: Multiple revision cycles for quality
- **Multi-Format Input**: Handles text, audio, and video seamlessly
- **Flexible LLM Backend**: Works with any supported LLM provider
- **Rich Output**: Professional book format with all required sections
- **State Management**: LangGraph handles complex workflow states
- **Quality Focus**: Prioritizes output quality over speed
