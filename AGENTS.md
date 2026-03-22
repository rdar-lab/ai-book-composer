# AGENTS.md — AI Book Composer

Read this file at the start of every session. It contains project-specific conventions, architectural context, and rules that must be followed.

---

## Mandatory Conventions

### Keep AGENTS.md Up to Date

After any change that affects architecture, agent logic, state fields, config schema, workflow edges, prompt keys, caching behaviour, or data flow — update the relevant section of this file in the same session. Do not defer this.

---

### Always Run Ruff Before Finishing

Run ruff on any Python files you change before considering a task complete.

```bash
cd /home/xmaster/ai-book-composer
ruff check src/ai_book_composer/<changed_file>.py
# or all at once:
ruff check .
```

---

### Run Tests Correctly

```bash
# Unit tests
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Single test file
uv run pytest tests/unit/test_executor.py -v
```

---

### Never Call `llm.invoke()` Directly from Agents

All LLM calls from agent code must go through `AgentBase._invoke_llm()` or `AgentBase._invoke_agent()`. These methods own retry logic (3 attempts, 60s wait), state-summary injection, and progress display callbacks. Calling `llm.invoke()` directly bypasses all of this.

---

### Never Set `iterations` to an Absolute Value

`AgentState.iterations` is defined with `Annotated[int, operator.add]` as its LangGraph reducer. Always return `{"iterations": 1}` from nodes that increment it. Returning `{"iterations": current + 1}` or any absolute value corrupts the counter across graph invocations.

The only place this is written is in `CriticAgent.critique()`:
```python
return {"critic_feedback": feedback, "quality_score": quality_score, "status": status, "iterations": 1}
```

---

### Never Parse Raw LLM Output with `json.loads()` Directly

Always use `extract_json_from_llm_response(text)` from `llm.py`. It handles markdown code fences (` ```json ``` `), bare JSON inside prose, and `<result>` XML blocks. Calling `json.loads()` on raw LLM output will fail in most real-world cases.

---

### Do Not Add Agent Logic to `workflow.py`

`workflow.py` exclusively wires nodes together. All agent behaviour belongs in agent files under `agents/`. The workflow file contains: node definitions that delegate to agents, conditional edge functions, and `_record_execution()`.

---

## Project Architecture

### What This System Does

Generates a structured DOCX book from a directory of raw source files (text, PDF, audio, video, images) using a multi-agent LangGraph pipeline. The entry point is a Click CLI. There is no web server, no database, and no frontend. Everything runs in a single Python process.

### Top-Level Structure

```
/home/xmaster/ai-book-composer/
├── src/ai_book_composer/
│   ├── cli.py                 # Click entry point
│   ├── config.py              # All config dataclasses + Settings loader + load_prompts()
│   ├── llm.py                 # LLM factory, agent invocation, ToolFixer, JSON extraction
│   ├── workflow.py            # LangGraph StateGraph — wires agents into a pipeline
│   ├── rag.py                 # RAGManager: ChromaDB in-memory vector store
│   ├── progress_display.py    # Rich terminal UI (no logic)
│   ├── parallel_utils.py      # execute_parallel() wrapping ThreadPoolExecutor
│   ├── logging_config.py      # Logging setup
│   └── agents/
│       ├── state.py           # AgentState TypedDict + create_initial_state()
│       ├── agent_base.py      # Base class: LLM calls, tools, state helpers
│       ├── preprocess_agent.py  # Phase 1: file ingestion, transcription, RAG init
│       ├── planner.py         # Phase 2: task planning
│       ├── executor.py        # Phase 3: chapter planning, generation, references
│       ├── decorator.py       # Phase 4: AI image placement
│       ├── critic.py          # Phase 5: quality scoring and approval
│       └── writer.py          # Phase 6: delegates to BookWriter
│   └── utils/
│       ├── file_utils.py      # All file I/O: read, transcribe, extract, cache
│       ├── book_writer.py     # BookWriter: assembles final DOCX
│       └── term_extraction.py # Key term extraction from gathered text
│
├── config.yaml                # Runtime configuration (consumed by Settings)
├── prompts.yaml               # All LLM prompts (consumed by load_prompts())
├── models.json                # Embedded Ollama model registry (repo_id + filename)
├── requirements.txt           # Pinned production dependencies
├── requirements.in            # Unpinned dependency sources
├── requirements-dev.txt       # Dev/test dependencies
├── .python-version            # Python version pin (pyenv)
│
├── run/                       # Runtime artifacts — not committed
│   ├── input/                 # Source files to process
│   ├── output/                # Generated books land here
│   ├── .cache/                # Per-run LLM output cache (see general.cache_dir)
│   └── logs/                  # Application logs
│
└── tests/
    ├── unit/
    └── integration/
```

---

## Running the CLI

```bash
python -m src.ai_book_composer.cli \
  --input-dir ./run/input \
  --output-dir ./run/output \
  --title "Book Title" \
  --author "Author Name" \
  --language en-US \
  --max-iterations 3 \
  --style-instructions "professional and concise" \
  --config ./config.yaml        # optional; uses hardcoded defaults if omitted
```

All `--` options except `--input-dir` and `--output-dir` fall back to `config.yaml` values when not provided on the CLI. The CLI writes the output DOCX to `<output-dir>/book.docx`.

---

## Shared State: AgentState

`agents/state.py` — every LangGraph node reads from and writes to this. Understand all fields before touching any agent.

```python
class AgentState(TypedDict):
    # Set by create_initial_state() from CLI args / workflow init
    input_directory: str
    output_directory: str
    language: str                          # e.g. "en-US"
    style_instructions: str                # e.g. "academic book" — empty string if not provided
    book_title: str
    book_author: str

    # Written by PreprocessAgent
    files: List[Dict[str, Any]]            # [{path, name, extension}, ...]
    images: List[Dict[str, Any]]           # [{path, filename, description}, ...]
                                           # Only populated if settings.book.decorate_with_images=True
    gathered_content: Dict[str, Any]       # {file_path: {type, name, path, content, summary}}
    rag_manager: Optional[Any]             # Live RAGManager instance; None if RAG disabled or failed
    key_terms: List[str]                   # Up to 100 key terms from term_extraction.py

    # Written by PlannerAgent
    plan: List[Dict[str, Any]]             # [{task, description, status}, ...]
    current_task_index: int                # Index of next task for ExecutorAgent to run

    # Built up across workflow nodes
    execution_history: List[Dict[str, Any]]  # [{node, status, task_index?, task_type?}, ...]

    # Written by ExecutorAgent
    chapter_list: List[Dict[str, Any]]     # [{number, title, description}, ...]
    chapters: List[Dict[str, Any]]         # [{number, title, content, images?: []}]
    references: List[str]                  # ["filename - Source file: path", ...]

    # Written by CriticAgent — return {"iterations": 1} not absolute value
    iterations: Annotated[int, operator.add]
    critic_feedback: Optional[str]
    quality_score: Optional[float]         # 0.0–1.0

    # Written by WriterAgent
    final_output_path: Optional[str]

    # Written at various phases
    status: str                            # "initialized" | "preprocessed" | "planned" |
                                           # "executing" | "execution_complete" | "approved" |
                                           # "needs_revision" | "book_generated" | "completed"
    error: Optional[str]
```

`create_initial_state()` sets `iterations=0` and all lists/dicts to empty. The `status` starts as `"initialized"`.

---

## Workflow Graph

`workflow.py` — `BookComposerWorkflow` builds a `StateGraph[AgentState]`.

### Node Sequence

```
START → preprocess → plan → execute → decorate → critique → write → finalize → END
```

### Conditional Edges

**After `execute`** — `_should_continue_execution()`:
```python
if current_task_index >= len(plan):
    return "decorate"
else:
    return "continue"   # routes back to "execute" node
```
`execute` calls itself repeatedly until all tasks are done. Each call processes one task and increments `current_task_index`.

**After `critique`** — `_should_revise()`:
```python
if status == "approved" or iterations >= self.max_iterations:
    return "write"
else:
    # Mutates state and settings directly:
    state['current_task_index'] = 0
    self.settings.book.use_cached_chapters_list = False
    self.settings.book.use_cached_chapters_content = False
    self.settings.book.use_cached_decorations = False
    # Resets all plan tasks to "pending"
    for task in state.get("plan", []):
        task["status"] = "pending"
    return "revise"     # routes back to "execute" node
```

### Node Retry Behaviour

Every node (except `decorate`) uses `tenacity.Retrying(stop=stop_after_attempt(3), wait=wait_fixed(60))`. A node failure that exhausts retries raises `RuntimeError` and stops the whole workflow.

`decorate` is different: if all 3 attempts fail, it logs a warning, returns `{"status": "decoration_failed"}`, and the workflow continues to `critique`. It never raises.

### Execution History

After every node, `_record_execution(state, node_name, status, step_index)` appends to `execution_history`. For `execute` nodes it also records `task_index` and `task_type` from the plan.

---

## Configuration

### Settings Class (`config.py`)

`Settings.__init__(config_path=None)` loads config in this order:
1. If `config_path` is `None` → use `_get_defaults()` (hardcoded Python dict)
2. If `config_path` is provided → load YAML file
3. Call `_replace_env_vars(data)` — substitutes `${VAR_NAME}` strings with `os.environ` values
4. Construct sub-config dataclass instances from the loaded dict

`Settings` is created once in `cli.py` and passed into `BookComposerWorkflow`, then forwarded to every agent. Never recreate or reload it inside agents.

### Config Dataclasses (all fields)

**`LLMConfig`** — `settings.llm`
| Field | Default | Description |
|-------|---------|-------------|
| `provider` | `"ollama_embedded"` | LLM provider key |
| `model` | `"llama-3.2-1b-instruct"` | Model name |
| `temperature` | `{planning: 0.3, execution: 0.85, critique: 0.2, decoration: 0.3}` | Per-phase temps |
| `static_plan` | `True` | If True, PlannerAgent skips LLM and uses hardcoded 3-task plan |
| `use_deep_agent` | `False` | Use `deepagents.create_deep_agent` instead of `langchain.create_agent` |
| `use_tool_fixer` | `False` | Wrap LLM in `ToolFixer` to fix XML-style tool calls |
| `agent_debug_mode` | `False` | Pass `debug=True` to agent creation |

**`BookConfig`** — `settings.book`
| Field | Default | Description |
|-------|---------|-------------|
| `output_language` | `"en-US"` | Default language (overridable by CLI) |
| `default_title` | `"Composed Book"` | Fallback book title |
| `default_author` | `"AI Book Composer"` | Fallback author |
| `quality_threshold` | `0.7` | Minimum score for CriticAgent approval |
| `max_iterations` | `3` | Max critique→execute revision cycles |
| `style_instructions` | `""` | Default style instructions |
| `min_words_per_chapter` | `500` | Chapter validation minimum; raises on failure → triggers 10-attempt retry |
| `use_cached_plan` | `True` | Cache PlannerAgent output |
| `use_cached_chapters_list` | `True` | Cache chapter list; set to False on revision |
| `use_cached_chapters_content` | `True` | Cache individual chapters; set to False on revision |
| `use_cached_decorations` | `True` | Cache decorator output; set to False on revision |
| `decorate_with_images` | `False` | If False, images are never gathered and decoration phase is a no-op |

**`RAGConfig`** — `settings.rag`
| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `True` | If False, RAG is skipped; agents cannot use `get_relevant_documents` |
| `embedding_model` | `"all-MiniLM-L6-v2"` | HuggingFace sentence-transformers model |
| `chunk_size` | `1000` | Characters per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |
| `max_allowed_distance` | `0.9` | Filter threshold; documents with distance > this are excluded |

**`WhisperConfig`** — `settings.whisper`
| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `"local"` | `"local"` or `"remote"` |
| `model_size` | `"base"` | `"base"`, `"small"`, `"medium"`, `"large"` |
| `remote.endpoint` | `"http://localhost:9000"` | Remote Whisper endpoint |
| `remote.api_key` | `None` | Optional auth |
| `local.device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `local.compute_type` | `"int8"` | Quantization type |

**`ImageProcessingConfig`** — `settings.image_processing`
| Field | Default | Description |
|-------|---------|-------------|
| `supported_formats` | `[jpg, jpeg, png, gif, bmp]` | Image file extensions |
| `extract_from_pdf` | `True` | Extract embedded images from PDFs |
| `max_image_size_mb` | `10` | Image file size limit |
| `max_images_per_chapter` | `5` | Decoration limit per chapter |

**`VisionModelConfig`** — `settings.vision_model`

Separate from the main LLM. Used only for image description in Preprocess. Default: `openai / gpt-4o-mini / temperature=0.3`.

**`ParallelConfig`** — `settings.parallel`
| Field | Default | Description |
|-------|---------|-------------|
| `parallel_execution` | `True` | Enable parallel processing |
| `parallel_workers` | `4` | Thread pool size (1–32) |

**`GeneralConfig`** — `settings.general`
| Field | Default | Description |
|-------|---------|-------------|
| `cache_dir` | `".cache"` | Directory for cache files (relative to cwd) |

**`LoggingConfig`** — `settings.logging`
| Field | Default | Description |
|-------|---------|-------------|
| `level` | `"INFO"` | Log level |
| `file` | `"logs/ai_book_composer.log"` | Log file path |
| `console_output` | `False` | Also print to console |

**`SecurityConfig`** — `settings.security`
| Field | Default | Description |
|-------|---------|-------------|
| `allow_directory_traversal` | `False` | Block `../` paths |
| `max_file_size_mb` | `500` | Max file size to read |

### Provider Configuration

API keys and provider-specific options live in `settings.providers` (a plain dict). Access via `settings.get_provider_config("openai")`. In config.yaml, these go under the `providers:` key. All api keys default to reading from environment variables:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  gemini:
    api_key: ${GOOGLE_API_KEY}
  azure:
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    deployment: ${AZURE_OPENAI_DEPLOYMENT}
  bedrock:
    region_name: ${AWS_REGION}
    aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}
    aws_session_token: ${AWS_SESSION_TOKEN}
  ollama:
    base_url: http://localhost:11434
  ollama_embedded:
    run_on_gpu: false
    internal:
      n_ctx: 131072
      n_threads: 4
      n_batch: 64
      verbose: false
```

---

## LLM Layer (`llm.py`)

### `get_llm(settings, temperature=0.7, model=None, provider=None) → BaseChatModel`

Factory for all LLM instances. `model` and `provider` override `settings.llm.model` and `settings.llm.provider` if provided.

| Provider key | LangChain class | Notes |
|---|---|---|
| `openai` | `ChatOpenAI` | |
| `gemini` | `ChatGoogleGenerativeAI` | |
| `azure` | `AzureChatOpenAI` | Needs deployment, api_key, endpoint |
| `anthropic` | `ChatAnthropic` | |
| `bedrock` | `ChatBedrockConverse` | Falls back to AWS credentials file if no explicit keys |
| `ollama` | `ChatOllama` | Needs external Ollama server running |
| `ollama_embedded` | `ChatLlamaCpp` (langchain_community) | Downloads GGUF from HuggingFace to `~/.cache/ai-book-composer/models/` on first use |

For `ollama_embedded`, `run_on_gpu=True` sets `n_gpu_layers=-1` (all layers on GPU); `False` sets `n_gpu_layers=0`.

### `invoke_agent(settings, model, system_prompt, user_prompt, tools, response_format, progress_callback)`

Multi-turn agentic call. Wraps with `ToolFixer` if `settings.llm.use_tool_fixer=True`. Uses `create_deep_agent` if `settings.llm.use_deep_agent=True`, otherwise `langchain.create_agent`. Sends messages as:
```python
{"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]}
```
Returns `(thought: str, action: str)` tuple.

### `invoke_llm(settings, model, system_prompt, user_prompt)`

Single-shot, no tools. Concatenates prompts as one string: `f'{system_prompt}\n{user_prompt}'` passed to `model.invoke()`. Returns `(thought: str, action: str)` tuple.

### `extract_json_from_llm_response(text) → Any`

Tries in order: (1) extract from ` ```json ``` ` block, (2) regex match for `[...]` or `{...}` structures, (3) parse the whole text. Raises `json.JSONDecodeError` if all fail.

### `_extract_thought_and_action(llm_response) → (thought, action)`

Handles three response forms:
1. `ThinkAndRespondFormat` Pydantic object → `.think` and `.result`
2. Text with `<think>...</think>` and `<result>...</result>` blocks → extracts inner content
3. Plain text → `thought=""`, `action=full_text`

Raises if `<think>` appears inside the extracted action (model did not follow format).

### `ToolFixer`

Active only when `settings.llm.use_tool_fixer=True`. Wraps a model to intercept XML-style `<tool_call>{...}</tool_call>` outputs and convert them to native LangChain `tool_calls`. Also deduplicates: if the same tool+args combo appears in history, it swaps the call to `system_notification` with a message telling the agent not to repeat it.

**History pruning** (inside `ToolFixer._prune_history()`):
- Keeps system message (index 0) intact
- Keeps last 4 message turns (`_KEEP_LAST_N_TURNS = 4`) intact
- Older `ToolMessage` objects → compressed to first 200 chars + length annotation
- Recent `ToolMessage` objects > 3000 chars → compressed to first 1000 chars

### `system_notification` Tool

Always included via `generate_default_tools()`. Used by `ToolFixer` to redirect duplicate tool calls. Agents should read it and adjust strategy — it is not a user-facing alert.

### `generate_default_tools() → [system_notification]`

Returns the base tool list included in every agent invocation.

---

## AgentBase (`agents/agent_base.py`)

All agents extend `AgentBase`. It holds `settings`, `prompts` (loaded from `prompts.yaml`), `llm_temperature`, and `self.state` (set at start of each agent call).

### Key Methods

**`_get_llm()`** — calls `get_llm(self.settings, temperature=self.llm_temperature)`. Called fresh each invocation (not cached).

**`_invoke_llm(system_prompt, user_prompt, include_agent_state=True)`**
- Retry: 3 attempts, 60s wait
- If `include_agent_state=True`: prepends `_get_agent_state_summary()` to system_prompt
- Calls `invoke_llm()` from `llm.py`
- Returns `action` string (discards `thought`)

**`_invoke_agent(system_prompt, user_prompt, state=None, custom_tools=None, include_agent_state=True, response_format=ThinkAndRespondFormat)`**
- Retry: 3 attempts, 60s wait
- If `include_agent_state=True`: appends state summary to system_prompt
- If `custom_tools` is None: calls `_generate_tools()` for default tool set
- Attaches `AgentProgressCallbackHandler` for Rich console output
- Returns `action` string

**`_generate_tools() → list[BaseTool]`**
Returns `generate_default_tools() + [get_relevant_documents_tool()]`. This is the default tool set given to all agents. Note: `get_file_content_tool()` is defined on `AgentBase` but is NOT in `_generate_tools()` — it is only used by `ExecutorAgent.get_custom_agent_tools()`.

**`get_relevant_documents_tool()`** — tool: `get_relevant_documents(query: str, num_results: int = 5)`
- Queries `self.state["rag_manager"]`
- `num_results` clamped to 1–10
- Returns `{query, num_results, results: [{rank, content, file_name, file_type, similarity_score, chunk_info}]}`
- Returns `{error: "RAG system not initialized"}` if `rag_manager` is None

**`get_file_content_tool()`** — tool: `get_file_content(file_name: str, start_char: int = 0, length: int = 5000)`
- Looks up `file_name` in `self.state["gathered_content"]` by filename (not full path)
- `length` clamped to max 10000
- Returns `{file_name, file_type, chunk, start_char, end_char, total_length, has_more}`

**`_get_files_summary(sample_size=100) → str`**
- If `gathered_content` has > 100 files: random sample of 100, with note about total
- Formats as JSON list of `{name, summary}` dicts
- Appends first 30 key_terms as comma-separated string

**`_get_agent_state_summary() → str`**
Compact state context injected into prompts. Contains:
- Plan steps: each step with `[STATUS]`, name, description, `<- CURRENT` marker
- Last 3 execution history entries
- `critic_feedback` (truncated to 200 chars at `MAX_CRITIC_FEEDBACK_LENGTH`)
- `iterations` count (only if > 0)
- `quality_score` as percentage (only if set)

---

## Agent: PreprocessAgent

**File:** `agents/preprocess_agent.py`
**Temperature:** not set (no LLM calls except summarization which uses `_invoke_llm`)
**State written:** `files`, `gathered_content`, `images`, `rag_manager`, `key_terms`, `status="preprocessed"`

### Execution Sequence in `preprocess()`

1. `gather_content()`:
   a. `list_files()` → calls `file_utils.list_input_files()`, stores in `state["files"]`
   b. `_gather_all_content(files)` → `execute_parallel(_process_single_file, files)` → stores in `gathered_content`
   c. `_summarize_all_files(gathered_content)` → `execute_parallel(_summerize_gathered_file, ...)`
   d. If `settings.book.decorate_with_images`: `_gather_images(files)` → `_describe_all_images(images)`

2. If `settings.rag.enabled`: `_initialize_rag()` → creates `RAGManager`, calls `ingest_documents(gathered_content)`

3. `_extract_key_terms()` → `extract_key_terms(all_content_strings, max_terms=100)`

### File Dispatch (`_process_single_file`)

| Extensions | Result type | Function |
|---|---|---|
| `.txt`, `.md`, `.rst`, `.docx`, `.rtf`, `.pdf` | `"text"` | `read_text_file()` |
| `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac` | `"audio_transcription"` | `read_audio_file()` |
| `.mp4`, `.avi`, `.mov`, `.mkv` | `"video_transcription"` | `read_video_file()` |
| anything else | `"unsupported"` | returns error content, `status="skipped"` |

Individual file failures are caught and stored with `status="error"` — they do not stop processing of other files.

### Summarization Logic (`_summerize_gathered_file`)

- Content < 2000 chars (`_MIN_LENGTH_FOR_SUMMARIZATION`): skip LLM, use content as summary
- Content >= 2000 chars: call `_invoke_llm()` with first 16384 chars (`_MAX_LENGTH_FOR_SUMMARIZATION`)
- Result cached by file path + language hash
- Stored in `gathered_file['summary']` truncated to 2000 chars (`_MAX_SUMMARY_LENGTH`)

### Image Handling

Only runs if `settings.book.decorate_with_images=True`. Collects images from:
1. `list_images()` — existing images in input directory
2. `extract_images_from_pdf()` — for each `.pdf` file, run in parallel

Image descriptions via `describe_image()` use the **vision model** (`settings.vision_model`), not the main LLM. Results are cached.

---

## Agent: PlannerAgent

**File:** `agents/planner.py`
**Temperature:** `settings.llm.temperature.get('planning', 0.3)`
**State written:** `plan`, `status="planned"`

### Behaviour

By default (`settings.llm.static_plan=True`): skips LLM entirely and returns a hardcoded 3-task plan:
```python
[
    {"task": "plan_chapters", "description": "Determine book structure and chapters", "status": "pending"},
    {"task": "generate_chapters", "description": "Write each chapter based on gathered content", "status": "pending"},
    {"task": "compile_references", "description": "Compile list of references", "status": "pending"}
]
```

If `settings.llm.static_plan=False`: calls `_invoke_agent()` with `planner.system_prompt` and `planner.user_prompt`. Result is cached to `planner_plan.json` (only if `settings.book.use_cached_plan=True`). The cache is NOT cleared on revision.

### Plan Validation (`_parse_plan`)

LLM plan must be a JSON array. Each task dict must have `task`, `description`, and `status` fields. Raises `ValueError` on any missing field.

---

## Agent: ExecutorAgent

**File:** `agents/executor.py`
**Temperature:** `settings.llm.temperature.get('execution', 0.85)`
**State written:** `chapter_list`, `chapters`, `references`, `current_task_index` (incremented by 1 each call)

### Task Dispatch (`execute()`)

Reads `current_task_index` from state, runs the corresponding task, increments index, returns updated state. Called once per task by the LangGraph loop.

| `task` value | Method | Retry |
|---|---|---|
| `"plan_chapters"` | `_plan_chapters_inner()` | 10 attempts |
| `"generate_chapters"` | `_generate_chapters_inner()` | — (inner wrapper retries 10×) |
| `"compile_references"` | `_compile_references_inner()` | — |
| any other value | `_custom_agent_task()` | 3 attempts (via `_invoke_agent`) |

### Task 1: Plan Chapters (`_plan_chapters_inner`)

Retry: **10 attempts** (tenacity on the method itself).

1. Check cache (`chapter_list.json`) if `settings.book.use_cached_chapters_list=True`
2. If no cache: `_plan_chapters_with_llm()` → `_invoke_agent()` with `executor.chapter_planning_system_prompt` + `executor.chapter_planning_user_prompt`
3. Parse response: `_parse_chapter_list()` — tries JSON first, then falls back to regex `r'^(?:Chapter\s+)?(\d+)[\.:]\s*(.*)'`
4. Validate: `_evaluate_chapter_list_quality()` — calls LLM with `chapter_list_critic_system_prompt`; rejects if response contains "reject", "revise", "needs improvement", or "not approve"; stores failure details in `self.previous_attempt_details` for next retry
5. If approved: cache result, clear `previous_attempt_details`
6. Minimum chapter count: `MIN_CHAPTER_COUNT = 3` — raises `ValueError` if fewer

### Task 2: Generate Chapters (`_generate_chapters_inner`)

Calls `execute_parallel(generate_chapter_wrapper, chapter_list)`. `fail_on_error=True` — any chapter failure aborts the whole task.

**`generate_chapter_wrapper(chapter_info)`** — retry: **10 attempts**:
1. Check cache (`chapter_{num}_content.txt`) if `use_cached_chapters_content=True`
2. If no cache: `_generate_chapter_with_llm()` → `_invoke_agent()` with `chapter_generation_system_prompt` + `chapter_generation_user_prompt`
3. Parse via `_parse_chapter_content_response()` — tries `extract_json_from_llm_response()`, falls back to raw text
4. Validate word count: if `< settings.book.min_words_per_chapter` (default 500) → stores failure details, raises to trigger retry
5. Validate content quality: `_evaluate_chapter_content_quality()` — sends first 10000 chars to LLM with `chapter_content_critic_system_prompt`; same approve/reject logic as chapter list evaluation
6. If approved: cache result

On retry, `self.previous_attempt_details` is injected into the generation prompt via `{previous_attempt_details}` in `chapter_generation_user_prompt`. This tells the LLM what was wrong with the prior attempt.

### Task 3: Compile References (`_compile_references_inner`)

No LLM call. Iterates `state["files"]` and formats each as: `"<file_name> - Source file: <file_path>"`.

### Custom Agent Tools (Executor-Specific)

`get_custom_agent_tools()` extends base tools with:
- `plan_chapters_tool()` — wraps `_plan_chapters_inner()` as a LangChain tool
- `generate_chapters_tool()` — wraps `_generate_chapters_inner()`
- `compile_references_tool()` — wraps `_compile_references_inner()`

These are only used by `_custom_agent_task()` for unknown task types.

---

## Agent: DecoratorAgent

**File:** `agents/decorator.py`
**Temperature:** `settings.llm.temperature.get('decoration', 0.3)`
**State written:** `chapters[i]["images"]` populated for each chapter

If `settings.book.decorate_with_images=False` or no images in state, the agent is a no-op (returns immediately).

For each chapter: sends chapter content preview + all available image descriptions to LLM with `image_placement_system_prompt`. LLM returns JSON list:
```json
[{"image_path": "...", "position": "start|middle|end", "reasoning": "..."}]
```
`position` must be one of `"start"`, `"middle"`, `"end"`. Capped at `settings.image_processing.max_images_per_chapter`. Result cached if `use_cached_decorations=True`.

---

## Agent: CriticAgent

**File:** `agents/critic.py`
**Temperature:** `settings.llm.temperature.get('critique', 0.2)`
**State written:** `critic_feedback`, `quality_score`, `status`, `iterations: 1`

### Behaviour

1. Summarize all chapters: `_summarize_chapters()` — first 1000 chars per chapter (`CHAPTER_PREVIEW_LENGTH`) + word count
2. Call `_invoke_agent()` with `critic.system_prompt` + `critic.user_prompt`
3. Parse via `_parse_critique()`:
   - `_extract_score()`: scans lines for "score" or "quality" keywords; handles 0–1, 0–10, 0–100 scales
   - `_is_approved()`: looks for "revise"/"needs work"/"improve" (reject) vs "approve"/"good"/"excellent" (approve)
4. Approval condition: `decision == "approve"` OR `quality_score >= quality_threshold`

Returns `{"iterations": 1}` always — the LangGraph `operator.add` reducer accumulates this.

If no chapters in state, returns `quality_score=0.0` and `status="needs_revision"` immediately.

---

## Agent: WriterAgent

**File:** `agents/writer.py`
**No LLM call.** Delegates entirely to `BookWriter`.

Calls `BookWriter.run(title, author, chapters, references, output_filename="book.docx")` and stores result path in `state["final_output_path"]`.

---

## BookWriter (`utils/book_writer.py`)

Builds the DOCX using `python-docx`.

### Document Structure

1. **Normal style**: Times New Roman 12pt (set on document default)
2. **Title page**:
   - Title: Arial Bold 24pt, centered
   - "By {author}": Arial 14pt, centered
   - Date: Arial 12pt `"%B %Y"` format, centered
3. **Table of Contents** (new page): Heading level 1 "Table of Contents", then `"Chapter {i}: {title}"` for each
4. **Chapters** (each on new page):
   - Heading level 1: `"Chapter {i}: {title}"`
   - `start_images` → content paragraphs (split on `\n\n`) → `middle_images` at `num_paragraphs // 2` index → `end_images`
5. **References** (new page, if any): Heading level 1 "References", then one paragraph per reference string

### Image Insertion (`_add_image_to_doc`)

- Width: 6 inches (aspect ratio preserved by python-docx)
- Caption: italic Arial 10pt `"Figure: {reasoning}"`, centered — only added if `reasoning` is non-empty
- Silent failure if path does not exist or image cannot be read

---

## RAG System (`rag.py`)

### `RAGManager`

Initialized once during Preprocess, stored in `AgentState.rag_manager`.

**Init (`_init_components`):**
- Embeddings: `HuggingFaceEmbeddings(model_name=..., device='cpu', normalize_embeddings=True)`
- Splitter: `RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""])`
- ChromaDB: `is_persistent=False, anonymized_telemetry=False` — fully in-memory, no files written
- Collection name: `"document_collection"`

**`ingest_documents(gathered_content)`:**
- Iterates all `gathered_content` values, skips entries with empty content
- Splits each document into chunks, adds `{file_name, file_type, chunk_index, total_chunks}` metadata
- Calls `vectorstore.add_texts(texts, metadatas)` in one batch

**`retrieve_relevant_documents(query, k=5, filter_dict=None)`:**
- Calls `vectorstore.similarity_search_with_score(query, k, filter)`
- Filters out results where `score > max_allowed_distance` (default 0.9)
- Returns `[{content, metadata, similarity_score}]`

**Thread safety:** ChromaDB in-memory supports concurrent reads. Do not call `ingest_documents()` from parallel threads — ingestion is done once, serially, before any parallel operations.

---

## Prompts (`prompts.yaml`)

Loaded once via `load_prompts()` in `config.py`. `load_prompts()` searches `prompts.yaml` in CWD first, then falls back to the package root. Returns a nested dict. Agents access prompts as `self.prompts['agent_name']['prompt_key']`.

### Prompt Keys by Agent

| Agent key | Prompt keys |
|---|---|
| `preprocessor` | `summarization_system_prompt`, `summarization_user_prompt`, `image_description_system_prompt` |
| `planner` | `system_prompt`, `user_prompt` |
| `executor` | `chapter_planning_system_prompt`, `chapter_planning_user_prompt`, `chapter_generation_system_prompt`, `chapter_generation_user_prompt`, `chapter_list_critic_system_prompt`, `chapter_list_critic_user_prompt`, `chapter_content_critic_system_prompt`, `chapter_content_critic_user_prompt`, `llm_agent_system_prompt`, `llm_agent_user_prompt` |
| `critic` | `system_prompt`, `user_prompt` |
| `decorator` | `image_placement_system_prompt`, `image_placement_user_prompt` |

### Template Variables

Prompts use Python `.format(**kwargs)` at call time. Missing variables raise `KeyError` at runtime, not load time.

| Variable | Used in | Source |
|---|---|---|
| `{language}` | all | `state.get("language", "en-US")` |
| `{style_instructions_section}` | planner, executor, critic, decorator | Built from `state.get("style_instructions", "")` |
| `{file_summary}` | planner, executor | `agent_base._get_files_summary()` |
| `{previous_attempt_details}` | executor chapter/list planning | `self.previous_attempt_details` string |
| `{chapter_number}`, `{title}`, `{description}` | executor chapter generation | Per-chapter from `chapter_list` |
| `{book_title}`, `{chapter_count}`, `{reference_count}`, `{chapter_summaries}` | critic | From state |
| `{file_name}`, `{file_content}` | preprocessor summarization | Per-file |
| `{state}`, `{current_task}` | executor custom agent task | Direct state dict |

Editing `prompts.yaml` takes effect on the next run — no restart needed. When adding a new template variable to a prompt, also add the corresponding `.format(key=value)` call at the agent's prompt call site.

---

## Parallel Execution (`parallel_utils.py`)

### `execute_parallel(settings, func, items, fail_on_error=False) → List[Any]`

- If `parallel_workers == 1` or `parallel_execution=False`: runs sequentially
- Otherwise: `ThreadPoolExecutor(max_workers=parallel_workers)`
- Results list is pre-allocated to maintain input order (not completion order)
- On item failure: if `fail_on_error=True` → re-raises immediately; else → stores `{"error": ..., "item": ...}` in results

**Used for** (all parallelized by default):

| Operation | `fail_on_error` |
|---|---|
| File reading in Preprocess | `False` |
| File summarization in Preprocess | `False` |
| Image extraction from PDFs | `False` |
| Image description (vision LLM) | `False` |
| Chapter generation | `True` — one chapter failure aborts all |

Chapter generation is the highest-value parallelism: ~4× speedup with 4 workers on a 20-chapter book.

---

## Caching System (`utils/file_utils.py`)

Cache functions: `get_cache_path(settings, key, prefix="", language=None)`, `read_cache(path) → str | None`, `write_cache(path, value: str)`.

Cache directory: `settings.general.cache_dir` (default `".cache"`).

`get_cache_path()` uses `hashlib` to generate a deterministic filename from the key. For file-based keys (paths), it hashes the absolute path. For string keys, it hashes the string directly. Adding a `language` suffix prevents cross-language cache collisions for transcriptions and summaries.

### What Is Cached

| Content | Cache key pattern | Cleared on revision |
|---|---|---|
| Audio/video transcriptions | file path + language | Never |
| File summaries | file path + language + `"summary_"` prefix | Never |
| Image descriptions | image path | Never |
| PlannerAgent plan | `"planner_plan.json"` | Never (plan stays same on revision) |
| Chapter list | `"chapter_list.json"` | Yes — `use_cached_chapters_list=False` |
| Chapter content | `"chapter_{num}_content.txt"` | Yes — `use_cached_chapters_content=False` |
| Decorator placements | (per-chapter key in decorator) | Yes — `use_cached_decorations=False` |

Revision clears caches by setting `settings.book.use_cached_*=False` in `_should_revise()`. This mutates the live `settings` object, so subsequent cache reads during the revision cycle return `None` and force regeneration.

---

## Data Flow: Complete Pipeline

```
CLI
  └─ Settings.load() + create_initial_state()
       │
       ▼
[PREPROCESS]
  list_input_files()
  parallel: _process_single_file() for all files
    ├─ text → read_text_file() → {type: "text", content}
    ├─ audio → read_audio_file() → Whisper → {type: "audio_transcription", content}
    └─ video → ffmpeg + Whisper → {type: "video_transcription", content}
  parallel: _summerize_gathered_file() for all files
    └─ LLM summarize if > 2000 chars, cached by path+language
  if decorate_with_images:
    list_images() + parallel extract_images_from_pdf()
    parallel: describe_image() via vision model, cached
  if rag.enabled:
    RAGManager.ingest_documents() → ChromaDB in-memory
  extract_key_terms() → up to 100 terms
  → state: files, gathered_content, images, rag_manager, key_terms

[PLAN]
  if static_plan=True: return hardcoded 3-task plan (no LLM)
  if static_plan=False: LLM → parse JSON plan → cache
  → state: plan=[{plan_chapters, generate_chapters, compile_references}]

[EXECUTE × 3]
  Task 1 — plan_chapters (retry 10×):
    LLM with chapter_planning prompts + get_relevant_documents tool
    → parse JSON chapter list → quality check LLM → cache
    → state: chapter_list

  Task 2 — generate_chapters:
    parallel (retry 10× each): _generate_chapter_content()
      LLM with chapter_generation prompts + get_relevant_documents tool
      → parse content → word count check → quality check LLM → cache
    → state: chapters

  Task 3 — compile_references (no LLM):
    iterate state["files"] → format strings
    → state: references

[DECORATE]
  if decorate_with_images and images exist:
    for each chapter: LLM decides image placement → cache
    → state: chapters[i]["images"] populated

[CRITIQUE]
  summarize chapters (first 1000 chars each)
  LLM with critic prompts
  parse score (0–1) + decision (approve/revise)
  if approved OR iterations >= max_iterations → status="approved"
  else → status="needs_revision", iterations += 1

  ┌─ "approved" → [WRITE]
  └─ "needs_revision" → [EXECUTE] (reset task_index=0, disable chapter caches)

[WRITE]
  BookWriter.run(title, author, chapters, references)
  DOCX structure: title page → ToC → chapters with images → references
  → state: final_output_path

[FINALIZE]
  Display completion stats → status="completed"
```

---

## Output Format Contracts

LLM responses must match these shapes. `extract_json_from_llm_response()` handles unwrapping from markdown/prose, but the inner structure must match exactly.

**Chapter list** (Executor Task 1):
```json
[
  {"number": 1, "title": "Chapter Title", "description": "What this chapter covers"},
  {"number": 2, "title": "...", "description": "..."}
]
```
Minimum `MIN_CHAPTER_COUNT = 3` entries. If JSON fails, falls back to line-by-line regex: `r'^(?:Chapter\s+)?(\d+)[\.:]\s*(.*)'`.

**Plan** (Planner, only if `static_plan=False`):
```json
[
  {"task": "plan_chapters", "description": "...", "status": "pending"},
  {"task": "generate_chapters", "description": "...", "status": "pending"},
  {"task": "compile_references", "description": "...", "status": "pending"}
]
```
All three fields (`task`, `description`, `status`) are required. Optional `files` field must be a list if present.

**Chapter content** (Executor Task 2):
Plain text prose. Minimum `settings.book.min_words_per_chapter` words (default 500). No required JSON structure — `_parse_chapter_content_response()` first tries `extract_json_from_llm_response()` expecting a string, then falls back to raw text.

**Decorator output**:
```json
[
  {"image_path": "/absolute/path/to/image.jpg", "position": "start", "reasoning": "Establishes visual context"},
  {"image_path": "...", "position": "middle", "reasoning": "..."}
]
```
`position` must be `"start"`, `"middle"`, or `"end"`. Enforced by `BookWriter` when categorising images.

**Critic approval** — parsed from free-form LLM text:
- Score: first line containing "score" or "quality" with a number (handles 0–1, 0–10, 0–100)
- Decision: presence of "revise"/"needs work"/"improve" → reject; "approve"/"good"/"excellent" → approve

---

## File Format Support

All file I/O goes through `utils/file_utils.py`. Adding a new format requires: (1) adding a case in `_process_single_file()` in `preprocess_agent.py`, (2) implementing the read function in `file_utils.py`.

| Category | Extensions | Processing |
|---|---|---|
| Plain text | `.txt`, `.md`, `.rst` | Direct read with encoding detection |
| Word | `.docx` | python-docx |
| PDF | `.pdf` | PyMuPDF / pypdf (text extraction) |
| RTF | `.rtf` | striprtf |
| Audio | `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac` | faster-whisper (local or remote) |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv` | ffmpeg-python → audio → faster-whisper |
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp` | Pillow + vision LLM for description |
| PDF images | extracted from `.pdf` | PyMuPDF image extraction → Pillow |

---

## Embedded Model Registry (`models.json`)

Used by `get_llm()` when `provider="ollama_embedded"`. Maps model name → `{repo_id, filename}` for HuggingFace download. Downloaded to `~/.cache/ai-book-composer/models/`.

Available models:
| Model name | Notes |
|---|---|
| `llama-3.2-1b-instruct` | Smallest — `llama-3.2-1b-instruct` is the config default |
| `llama-3.2-3b-instruct` | Balanced |
| `llama-3.1-8b-instruct` | Best local quality; used in runtime defaults |
| `tinyllama` | Minimal footprint |
| `deepseek-r1-llama-8b` | Reasoning distilled |
| `deepseek-r1-qwen-7b` | Qwen-based |
| `qwen2.5-7b-instruct` | Alibaba |

All GGUF format from `bartowski/*` or similar HuggingFace repos.

---

## Key Invariants — Do Not Violate

1. **`iterations` is a reducer** — return `{"iterations": 1}` from `CriticAgent.critique()`. Never set to an absolute value.

2. **LLM calls from agents must go through `AgentBase._invoke_llm()` or `_invoke_agent()`** — these own retries, state injection, and callbacks.

3. **RAG ingestion is single-threaded** — `ingest_documents()` runs once in Preprocess before any parallel work. Do not call it from parallel threads.

4. **`extract_json_from_llm_response()` for all JSON parsing** — never `json.loads()` on raw LLM output.

5. **No agent logic in `workflow.py`** — workflow wires nodes only.

6. **`_should_revise()` mutates both `state` and `settings` directly** — it sets `state['current_task_index'] = 0` and flips cache flags on `settings.book`. This is intentional and must not be moved to the agent layer.

7. **Chapter generation retries 10× at the method level** — independent of the 3× node-level retry in `workflow.py`. These are different failure modes (content quality vs. LLM API errors).

8. **`static_plan=True` by default** — the PlannerAgent never calls an LLM unless explicitly configured. The plan is always the same 3 tasks. Don't add logic that assumes the plan can be arbitrary without checking this flag.

9. **`decorate_with_images=False` by default** — image gathering, description, and decoration are completely skipped unless this is True. Do not add code that assumes images are always populated.

10. **Decoration failure is non-fatal** — the `decorate` node returns `{"status": "decoration_failed"}` without raising. The workflow continues to `critique`. Never change this to raise.

---

## Non-Obvious Behaviours

- **`execute` calls itself repeatedly.** It is designed to be called once per task, with `current_task_index` as the cursor. The LangGraph conditional edge loops it back. Do not attempt to loop over tasks inside `execute()`.

- **`self.previous_attempt_details` is instance state on ExecutorAgent.** It is reset to `""` at the start of each `execute()` call and populated on validation failure inside `_capture_attempt_failure()`. It persists across the 10 retry attempts of a single task. It is cleared on success.

- **`invoke_llm()` concatenates prompts as a single string.** `f'{system_prompt}\n{user_prompt}'` is passed as one argument to `model.invoke()`. This is different from `invoke_agent()` which uses separate `SystemMessage` / `HumanMessage` objects.

- **The vision model is configured separately** from the main LLM (`settings.vision_model` vs `settings.llm`). `describe_image()` always uses the vision model, regardless of the main provider.

- **`_get_files_summary()` random-samples** if > 100 files. The sample changes between calls (uses `random.sample`). For prompts that need deterministic content, use the cache.

- **`_get_agent_state_summary()` truncates `critic_feedback` to 200 characters.** The full feedback is still in state and used by the critic prompt injection; only the state summary is truncated to keep prompts small.

- **History pruning in `ToolFixer._prune_history()`** only applies when `use_tool_fixer=True` (default False). When disabled, agents receive the full message history — context overflow is the agent's problem. Enable `use_tool_fixer` for providers that output XML-style tool calls (some Ollama models).

- **`_should_revise()` resets all plan tasks to `"pending"`.** On revision, `current_task_index` resets to 0 and all task `status` fields go back to `"pending"`. Execution history is preserved (not cleared) — agents can see the prior run's history in `_get_agent_state_summary()`.

- **Summarization skips short files.** Files with content under 2000 characters are stored as-is in `gathered_file['summary']` without an LLM call. Their `summary` field equals their full content (but still truncated to 2000 chars when stored).

- **`compile_references` does not read file content.** It iterates `state["files"]` (the list from `list_input_files`) and formats file path strings. It does not touch `gathered_content`. References are therefore always the complete file list, not filtered by what was used in generation.
