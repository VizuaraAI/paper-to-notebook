#!/usr/bin/env python3
"""
Paper to Notebook - Backend API
Single-file FastAPI application for deploying on Railway.
Converts research paper PDFs into executable Jupyter notebooks using LLM via OpenRouter.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
import time
import uuid
from typing import Callable, Optional

import httpx
import nbformat
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from openai import OpenAI

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "google/gemini-2.5-pro"

MAX_TOKENS_ANALYSIS = 12288
MAX_TOKENS_DESIGN = 12288
MAX_TOKENS_GENERATE = 65536
MAX_TOKENS_VALIDATE = 65536

MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]

MAX_PDF_SIZE_MB = 30
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", str(MAX_PDF_SIZE_MB)))

# Approximate char limit for paper text to avoid exceeding model context windows.
# ~80k chars ≈ ~20k tokens, leaving room for prompts + output within 128k context models.
MAX_PAPER_TEXT_CHARS = 80_000

# In-memory cache for /api/models
_models_cache: dict = {"data": None, "expires": 0}

# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = (
    "You are an expert research engineer and educator who converts academic papers "
    "into runnable, educational Python notebooks. You adapt your approach to the "
    "paper's domain: PyTorch for deep learning, NumPy/SciPy for algorithms and math, "
    "simulations for systems papers, etc. Your priorities:\n"
    "1. Faithfully replicate the paper's core ideas and algorithms\n"
    "2. Write complete, working code — no placeholders, no TODOs, no 'pass'\n"
    "3. Everything runs on a standard laptop (CPU only, no GPU, no large downloads)\n"
    "4. Explain every step clearly for someone learning the topic\n"
    "5. Include visualizations that reveal what the code is doing\n"
    "6. All output must be valid JSON — escape newlines as \\n and quotes as \\\" in strings"
)

ANALYSIS_PROMPT = """\
Read this research paper carefully and extract a structured analysis.

IMPORTANT: The paper may be about ANY topic — deep learning, classical algorithms, \
systems, mathematics, optimization, robotics, etc. Extract what is relevant and use \
null or empty arrays for fields that don't apply to this paper's domain.

Return a JSON object with these fields:

{
  "title": "Full paper title",
  "authors": "Author list as a single string",
  "paper_type": "One of: deep_learning, machine_learning, reinforcement_learning, nlp, computer_vision, optimization, algorithms, systems, theoretical, survey, other",
  "abstract_summary": "2-3 sentence plain English summary accessible to a non-expert",
  "problem_statement": "What problem does the paper solve? (2-3 sentences, avoid jargon)",
  "key_insight": "The core innovation in one sentence",
  "algorithms": [
    {
      "name": "Algorithm or method name",
      "description": "What it does in plain English",
      "inputs": ["list of inputs with types/shapes where applicable"],
      "outputs": ["list of outputs with types/shapes where applicable"],
      "steps": ["ordered list of DETAILED steps — math operations, loss functions, gradient updates, pseudocode logic"],
      "is_core": true,
      "equations": ["key equations in LaTeX or descriptive form"]
    }
  ],
  "baselines": [
    {
      "name": "Baseline method name",
      "description": "How it works — loss function, training procedure if applicable"
    }
  ],
  "evaluation_metrics": ["metrics used to evaluate, with formulas if available"],
  "key_equations": ["ALL important equations described precisely — include variable definitions"],
  "model_architecture": {
    "type": "Transformer/CNN/RNN/GNN/MLP/None/etc. (use None for non-ML papers)",
    "key_layers": ["layer types used, or empty for non-ML"],
    "dimensions": "hidden dim, num heads, num layers from paper, or null",
    "special_features": "non-standard architectural choices, or null"
  },
  "dataset": {
    "name": "Dataset name, or null if not applicable",
    "description": "Brief description",
    "preprocessing": "Special preprocessing steps"
  },
  "research_field": "2-4 word field name (e.g., 'Natural Language Processing', 'Graph Algorithms', 'Distributed Systems')",
  "key_contributions": ["Contribution in 4-7 words", "...", "..."],
  "implementation_approach": "Suggest what Python libraries and approach best demonstrate this paper (e.g., 'PyTorch neural network with synthetic data', 'NumPy matrix operations with visualization', 'NetworkX graph algorithm on generated graphs', 'simulation with matplotlib animation')"
}

RULES:
- Be exhaustive on algorithms, equations, and step-by-step details
- For non-ML papers, set model_architecture.type to "None" and leave ML-specific fields as null/empty
- Do NOT hallucinate — if the paper doesn't mention baselines or datasets, use empty arrays/null
- Return ONLY valid JSON — no trailing commas, no comments, no text outside the JSON
"""

DESIGN_PROMPT_TEMPLATE = """\
Design a **toy implementation plan** for a Jupyter notebook that demonstrates this paper's core ideas with working Python code.

**Paper Analysis:**
```json
{analysis_json}
```

Adapt the plan to the paper's domain:
- **Deep learning / ML papers**: Use PyTorch with small models, synthetic data, actual training loops
- **Algorithm papers**: Use NumPy/SciPy, implement the algorithm on toy inputs, benchmark against a naive approach
- **Systems papers**: Simulate the system behavior, measure and plot performance characteristics
- **Math / theoretical papers**: Implement key results numerically, visualize theorems and bounds
- **Optimization papers**: Implement the optimizer, show convergence on standard test functions

Return a JSON object:

{{
  "notebook_title": "Clear, descriptive title for the notebook",
  "pip_dependencies": ["list", "of", "pip", "packages", "needed"],
  "model_architecture": {{
    "type": "What we're building (e.g., 'Transformer', 'CNN', 'Graph Algorithm', 'Optimizer', 'Simulation', 'None')",
    "embed_dim": null,
    "num_layers": null,
    "num_heads": null,
    "description": "Brief description of the implementation architecture and key parameters — use small/toy values"
  }},
  "data_plan": {{
    "description": "What data to use — MUST be synthetic or generated in code, no external downloads",
    "size": "e.g., 500 samples — keep small for fast execution",
    "generation_method": "How to create it"
  }},
  "execution_plan": {{
    "description": "How to run the main experiment",
    "iterations": "number of epochs/steps/iterations — keep small enough to finish in under 2 minutes",
    "what_to_measure": "What metrics or outputs to track"
  }},
  "sections": [
    {{
      "title": "Section title",
      "cell_types": "markdown | code | both",
      "description": "What this section covers and why it matters",
      "key_elements": ["specific things to implement in this section"]
    }}
  ],
  "comparisons": [
    {{
      "name": "What to compare (e.g., 'Baseline vs Paper Method', 'Before vs After Optimization')",
      "metric": "How to measure the comparison"
    }}
  ],
  "visualizations": [
    {{
      "type": "line plot / bar chart / heatmap / scatter / animation / etc.",
      "description": "What it shows and what insight it provides"
    }}
  ]
}}

CONSTRAINTS:
- Only use pip-installable packages (no custom C extensions, no system-level dependencies)
- No external data downloads — generate ALL data synthetically or use tiny built-in datasets (e.g., sklearn.datasets)
- Everything must run on CPU in under 2 minutes total
- Plan 15-25 notebook cells for a thorough but focused implementation
- Sections should build on each other incrementally — each cell should work given all prior cells
- Always include: (1) overview/intro, (2) imports & setup, (3) core implementation, (4) experiment/demo, (5) visualization, (6) summary
- Return ONLY valid JSON — no trailing commas, no comments
"""

GENERATE_PROMPT_TEMPLATE = """\
Generate a **complete, runnable Jupyter notebook** as a JSON array of cells.

**Paper Analysis:**
```json
{analysis_json}
```

**Implementation Plan:**
```json
{design_json}
```

Follow the sections defined in the implementation plan above. For each planned section, \
create one or more notebook cells (markdown for explanations, code for implementation).

Each cell in the JSON array:
```json
{{
  "cell_type": "code" or "markdown",
  "source": "cell content as a single string"
}}
```

**CODE REQUIREMENTS:**
- Complete, working Python — NO placeholders, NO "# TODO", NO "pass", NO "..." stubs
- Every variable defined before use, every function fully implemented
- Only use packages listed in the plan's pip_dependencies
- Set random seeds early (random.seed, np.random.seed, torch.manual_seed if using PyTorch) for reproducibility
- Include print() statements that show intermediate results, shapes, and progress
- Add inline comments explaining non-obvious logic
- CPU only — no .cuda(), no .to('cuda'), no GPU device transfers
- No external file/data downloads — generate everything in code
- Keep data sizes and iteration counts small so the notebook runs in under 2 minutes

**MARKDOWN REQUIREMENTS:**
- Explain WHAT the code does and WHY — not just describe it
- Use analogies and plain English intuition before introducing math
- Reference the original paper's contributions where relevant
- Use LaTeX ($...$) for equations when helpful

**NOTEBOOK FLOW:**
- Start: title cell (paper name, authors, what this notebook demonstrates)
- Then: imports and setup (all imports in one cell, seeds, hyperparameters)
- Build incrementally: data preparation → core implementation → run experiment → evaluate → visualize
- End: summary of results, what we learned, ideas for extending this work

**JSON FORMATTING — CRITICAL, ERRORS HERE WILL BREAK PARSING:**
- Code newlines MUST be \\n (escaped newline), NOT literal line breaks inside the JSON string
- Double quotes in code MUST be \\" (escaped quote)
- Backslashes in code MUST be \\\\ (escaped backslash)
- No trailing commas after the last element in arrays or objects
- The output must be a single valid JSON array — no text before [ or after ]

Return ONLY the JSON array of cells.
"""

VALIDATE_PROMPT_TEMPLATE = """\
Validate and repair these Jupyter notebook cells. The notebook implements a research paper.

**Cells:**
```json
{cells_json}
```

**Check for ALL of these issues and fix them:**

1. **Import errors** — every package/module is imported before use, imports are in the first code cell
2. **Undefined variables** — every variable is defined before it's referenced, across all cells in order
3. **Syntax errors** — all Python code is syntactically valid (check quotes, brackets, colons, indentation)
4. **Logical flow** — cells produce correct results when run top-to-bottom sequentially
5. **No placeholders** — no "# TODO", "pass", or "..." standing in for real code
6. **Complete functions** — every function/class has a full working implementation, not stubs
7. **Meaningful output** — training loops print loss that should decrease; experiments compute and display results
8. **Package availability** — only standard pip packages (torch, numpy, matplotlib, scikit-learn, scipy, networkx, etc.)
9. **CPU compatibility** — no .cuda(), .to('cuda'), or GPU-specific code anywhere
10. **Reproducibility** — random seeds are set before any random operations
11. **No downloads** — no urllib, requests, wget, or any external data fetching in code cells
12. **Correct math** — loss functions, gradient computations, and algorithm steps match standard implementations

**Fix every issue found. If cells are already correct, return them unchanged.**

Return the corrected cells as a JSON array:
```json
[
  {{
    "cell_type": "code" or "markdown",
    "source": "corrected cell content"
  }}
]
```

**JSON FORMATTING:**
- Code newlines as \\n, quotes as \\", backslashes as \\\\
- No trailing commas
- Return ONLY the JSON array — no text before [ or after ]
"""

# ============================================================================
# LLM UTILITIES
# ============================================================================

def _get_api_key(api_key: str | None = None) -> str:
    return api_key or os.environ.get("OPENROUTER_API_KEY") or ""


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
    full_text = "\n\n".join(text_parts)
    if not full_text.strip():
        raise ValueError("Could not extract text from PDF. The PDF may be image-based or corrupted.")
    return full_text


def call_llm(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 8192,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Make an OpenRouter API call and return the text response."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_get_api_key(api_key),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if on_thinking:
        full_text = ""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )
        chunk_buffer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_text += delta.content
                chunk_buffer += delta.content
                if len(chunk_buffer) >= 200:
                    on_thinking(f"Processing... ({len(full_text)} chars generated)")
                    chunk_buffer = ""
        return full_text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""


def call_llm_with_retry(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 8192,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Call OpenRouter API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_llm(system_prompt, user_content, max_tokens, model, api_key, on_thinking)
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["invalid api key", "api_key_invalid", "unauthorized", "401", "authentication"]):
                raise ValueError("Invalid API key. Please check your OpenRouter API key and try again.")
            if any(kw in error_str for kw in ["429", "rate", "500", "503", "overloaded", "unavailable"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def parse_llm_json(raw_text: str, step_name: str, model: str, api_key: str | None = None) -> dict | list:
    """Parse JSON from LLM response, with cleanup and one repair attempt."""
    text = raw_text.strip()

    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed in {step_name}. Attempting repair...")
        repair_prompt = (
            f"The following text was supposed to be valid JSON but has a syntax error:\n\n"
            f"{text[:4000]}\n\n"
            f"Error: {e}\n\n"
            f"Return ONLY the corrected valid JSON, nothing else."
        )
        repaired = call_llm_with_retry(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=repair_prompt,
            max_tokens=max(len(text) // 2, 4096),
            model=model,
            api_key=api_key,
        )
        repaired = repaired.strip()
        if repaired.startswith("```"):
            repaired = repaired.split("\n", 1)[1]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        return json.loads(repaired.strip())

# ============================================================================
# NOTEBOOK BUILDER
# ============================================================================

def build_notebook(cells_json: list) -> nbformat.NotebookNode:
    """Convert a list of cell dicts into a proper .ipynb notebook."""
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.9",
    }

    for cell_data in cells_json:
        cell_type = cell_data["cell_type"]
        source = cell_data["source"]
        if cell_type == "markdown":
            nb.cells.append(new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(new_code_cell(source))
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    return nb


def nb_to_bytes(nb: nbformat.NotebookNode) -> bytes:
    buffer = io.StringIO()
    nbformat.write(nb, buffer)
    return buffer.getvalue().encode("utf-8")


def cells_to_bytes(cells: list) -> bytes:
    return nb_to_bytes(build_notebook(cells))

# ============================================================================
# PIPELINE
# ============================================================================

ProgressCallback = Callable[[int, str, str, Optional[dict]], None]
ThinkingCallback = Callable[[str], None]


def _clean_metric(m: str) -> str:
    """Strip formula parts (anything after = or parens) from a metric name."""
    return re.split(r'\s*[=(]', m)[0].strip().rstrip(',')


def run_pipeline(
    pdf_bytes: bytes,
    model: str = DEFAULT_MODEL,
    on_progress: Optional[ProgressCallback] = None,
    api_key: Optional[str] = None,
    on_thinking: Optional[ThinkingCallback] = None,
    model_max_output: int = 0,
) -> bytes:
    """Run the full pipeline on PDF bytes, returning .ipynb bytes."""

    def _notify(step: int, name: str, detail: str = "", extra: Optional[dict] = None):
        if on_progress:
            on_progress(step, name, detail, extra)

    def _clamp(desired: int) -> int:
        """Clamp max_tokens to model's limit if known."""
        if model_max_output > 0:
            return min(desired, model_max_output)
        return desired

    paper_text = extract_pdf_text(pdf_bytes)

    # Truncate to avoid exceeding model context windows
    if len(paper_text) > MAX_PAPER_TEXT_CHARS:
        paper_text = paper_text[:MAX_PAPER_TEXT_CHARS] + "\n\n[... truncated due to length ...]"

    # Step 1: Paper Analysis
    _notify(1, "Analyzing paper", "Reading PDF and extracting structure...")
    analysis_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Here is the full text of the research paper:\n\n{paper_text}\n\n{ANALYSIS_PROMPT}",
        max_tokens=_clamp(MAX_TOKENS_ANALYSIS),
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    analysis = parse_llm_json(analysis_raw, "paper_analysis", model, api_key=api_key)
    title = analysis.get("title", "Unknown Paper")
    num_algos = len(analysis.get("algorithms", []))
    raw_metrics = analysis.get("evaluation_metrics", [])
    clean_metrics = [_clean_metric(m) for m in raw_metrics[:4] if m and _clean_metric(m)]

    _notify(1, "Analyzing paper", f"Found: {title}", {
        "type": "analysis",
        "title": title,
        "algorithms": num_algos,
        "algorithm_names": [a.get("name", "") for a in analysis.get("algorithms", [])[:4] if a.get("name")],
        "insight": analysis.get("key_insight", ""),
        "problem": analysis.get("problem_statement", ""),
        "abstract_summary": analysis.get("abstract_summary", ""),
        "model_type": analysis.get("model_architecture", {}).get("type", ""),
        "authors": analysis.get("authors", ""),
        "research_field": analysis.get("research_field", ""),
        "key_contributions": analysis.get("key_contributions", [])[:3],
        "metrics": clean_metrics,
        "dataset_name": analysis.get("dataset", {}).get("name", ""),
        "key_layers": analysis.get("model_architecture", {}).get("key_layers", [])[:4],
        "baseline_names": [b.get("name", "") for b in analysis.get("baselines", [])[:3] if b.get("name")],
        "paper_type": analysis.get("paper_type", ""),
        "implementation_approach": analysis.get("implementation_approach", ""),
    })

    # Step 2: Design Plan
    _notify(2, "Designing implementation", "Planning architecture and notebook structure...")
    design_prompt = DESIGN_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2)
    )
    design_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Here is the full text of the research paper:\n\n{paper_text}\n\n{design_prompt}",
        max_tokens=_clamp(MAX_TOKENS_DESIGN),
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    design = parse_llm_json(design_raw, "toy_design", model, api_key=api_key)
    arch = design.get("model_architecture", {})
    num_sections = len(design.get("sections", []))
    _notify(2, "Designing implementation", "Implementation planned", {
        "type": "design",
        "notebook_title": design.get("notebook_title", ""),
        "model_type": arch.get("type", "") or arch.get("description", ""),
        "embed_dim": arch.get("embed_dim", ""),
        "num_layers": arch.get("num_layers", ""),
        "num_heads": arch.get("num_heads", ""),
        "num_sections": num_sections,
        "pip_dependencies": design.get("pip_dependencies", []),
    })

    # Step 3: Generate Notebook Cells
    _notify(3, "Generating notebook", "Writing code and explanations...")
    generate_prompt = GENERATE_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2),
        design_json=json.dumps(design, indent=2),
    )
    cells_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Here is the full text of the research paper:\n\n{paper_text}\n\n{generate_prompt}",
        max_tokens=_clamp(MAX_TOKENS_GENERATE),
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    cells = parse_llm_json(cells_raw, "generate_cells", model, api_key=api_key)
    num_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    previews = [{"type": c.get("cell_type", "code"), "preview": c.get("source", "")[:300]} for c in cells]

    draft_bytes = cells_to_bytes(cells)
    _notify(3, "Generating notebook", f"Generated {num_cells} cells ({code_cells} code)", {
        "type": "cells_generated",
        "num_cells": num_cells,
        "code_cells": code_cells,
        "previews": previews,
        "draft_bytes": draft_bytes,
    })

    # Step 4: Validate & Repair
    _notify(4, "Validating code", "LLM reviewing for errors...")
    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(
        cells_json=json.dumps(cells, indent=2)
    )
    validated_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=validate_prompt,
        max_tokens=_clamp(MAX_TOKENS_VALIDATE),
        model=model,
        api_key=api_key,
        on_thinking=on_thinking,
    )
    validated_cells = parse_llm_json(validated_raw, "validate", model, api_key=api_key)
    _notify(4, "Validating code", "Validation complete")

    return nb_to_bytes(build_notebook(validated_cells))

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Paper to Notebook API", version="2.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.mkdtemp(prefix="paper2nb_")
_generation_semaphore = asyncio.Semaphore(3)


# ============================================================================
# SHARED SSE STREAM
# ============================================================================

async def _stream_pipeline(pdf_bytes: bytes, model: str, api_key: str, job_id: str, model_max_output: int = 0):
    """Shared SSE generator that runs the pipeline and streams progress events."""
    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue = asyncio.Queue()
    draft_id = job_id + "_draft"

    def on_progress(step: int, name: str, detail: str, extra: dict = None):
        asyncio.run_coroutine_threadsafe(
            progress_queue.put(("progress", step, name, detail, extra)), loop,
        )

    def on_thinking(text: str):
        asyncio.run_coroutine_threadsafe(
            progress_queue.put(("thinking", text)), loop,
        )

    async def run_in_thread():
        async with _generation_semaphore:
            return await loop.run_in_executor(
                None,
                lambda: run_pipeline(pdf_bytes, model, on_progress, api_key=api_key, on_thinking=on_thinking, model_max_output=model_max_output),
            )

    task = asyncio.create_task(run_in_thread())

    while not task.done():
        try:
            event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue

        if event[0] == "thinking":
            yield f"event: thinking\ndata: {json.dumps({'text': event[1]})}\n\n"
            continue

        _, step, name, detail, extra = event

        if extra and "draft_bytes" in extra:
            draft_data = extra.pop("draft_bytes")
            draft_path = os.path.join(TEMP_DIR, f"{draft_id}.ipynb")
            with open(draft_path, "wb") as f:
                f.write(draft_data)
            yield f"event: progress\ndata: {json.dumps({'step': step, 'name': name, 'detail': detail, 'extra': extra})}\n\n"
            yield f"event: draft_ready\ndata: {json.dumps({'job_id': draft_id, 'size_kb': len(draft_data) // 1024})}\n\n"
        else:
            data = {"step": step, "name": name, "detail": detail}
            if extra:
                data["extra"] = extra
            yield f"event: progress\ndata: {json.dumps(data)}\n\n"

    # Drain remaining queued events
    while not progress_queue.empty():
        event = await progress_queue.get()
        if event[0] == "thinking":
            continue
        _, step, name, detail, extra = event
        if extra and "draft_bytes" in extra:
            extra.pop("draft_bytes")
        data = {"step": step, "name": name, "detail": detail}
        if extra:
            data["extra"] = extra
        yield f"event: progress\ndata: {json.dumps(data)}\n\n"

    # Send result or error to the client
    try:
        notebook_bytes = task.result()
    except Exception as e:
        print(f"Pipeline error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return

    output_path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")
    with open(output_path, "wb") as f:
        f.write(notebook_bytes)
    yield f"event: complete\ndata: {json.dumps({'job_id': job_id, 'size_kb': len(notebook_bytes) // 1024})}\n\n"


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Paper to Notebook API",
        "version": "2.0",
        "endpoints": {"generate": "/api/generate", "download": "/api/download/{job_id}", "health": "/health"},
    }


@app.get("/api/models")
async def list_models():
    """Fetch available models from OpenRouter with 5-minute in-memory cache."""
    now = time.time()
    if _models_cache["data"] and now < _models_cache["expires"]:
        return _models_cache["data"]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://openrouter.ai/api/v1/models", timeout=10.0)
            resp.raise_for_status()
            all_models = resp.json().get("data", [])
    except Exception:
        # Return stale cache on failure, or empty list if no cache
        if _models_cache["data"]:
            return _models_cache["data"]
        return {"models": [], "default": DEFAULT_MODEL}

    models = []
    for m in all_models:
        output_mods = m.get("architecture", {}).get("output_modalities", [])
        if "text" not in output_mods:
            continue
        ctx = m.get("context_length", 0)
        if ctx < 16000:
            continue
        max_out = m.get("top_provider", {}).get("max_completion_tokens") or 0
        if max_out < 16384:
            continue
        model_id = m.get("id", "")
        if any(skip in model_id for skip in ["/image", "tts", "embed", "whisper", "vision-only"]):
            continue

        provider = model_id.split("/")[0] if "/" in model_id else "other"
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", 0) or 0)

        models.append({
            "id": model_id,
            "name": m.get("name") or model_id,
            "provider": provider,
            "context_length": ctx,
            "max_output": max_out,
            "price_per_1m_tokens": round(prompt_price * 1_000_000, 2),
        })

    models.sort(key=lambda x: (x["provider"], x["name"]))
    result = {"models": models, "default": DEFAULT_MODEL}
    _models_cache["data"] = result
    _models_cache["expires"] = now + 300  # 5 minute TTL
    return result


@app.post("/api/generate")
async def generate(request: Request, file: UploadFile = File(...), api_key: str = Form(...), model: str = Form(DEFAULT_MODEL), model_max_output: int = Form(0)):
    """Generate notebook from uploaded PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    api_key = api_key.strip()
    if not api_key:
        raise HTTPException(400, "OpenRouter API key is required")

    pdf_bytes = await file.read()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"PDF too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")

    job_id = uuid.uuid4().hex[:12]
    return StreamingResponse(
        _stream_pipeline(pdf_bytes, model, api_key, job_id, model_max_output),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate-from-arxiv")
async def generate_from_arxiv(request: Request, arxiv_url: str = Form(...), api_key: str = Form(...), model: str = Form(DEFAULT_MODEL), model_max_output: int = Form(0)):
    """Generate notebook from arXiv URL."""
    api_key = api_key.strip()
    if not api_key:
        raise HTTPException(400, "OpenRouter API key is required")

    match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)', arxiv_url)
    if not match:
        raise HTTPException(400, "Invalid arXiv URL. Expected format: https://arxiv.org/abs/XXXX.XXXXX")

    paper_id = match.group(1)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        print(f"Downloading PDF from arXiv: {pdf_url}")
        response = await client.get(pdf_url, timeout=30.0)
        if response.status_code != 200:
            raise HTTPException(500, f"Failed to download PDF from arXiv: {response.status_code}")
        pdf_bytes = response.content

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"PDF too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")
    print(f"Downloaded PDF: {size_mb:.1f} MB")

    job_id = uuid.uuid4().hex[:12]
    return StreamingResponse(
        _stream_pipeline(pdf_bytes, model, api_key, job_id, model_max_output),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    """Download generated notebook by job ID."""
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(400, "Invalid job ID")
    path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")
    if not os.path.exists(path):
        raise HTTPException(404, "Notebook not found or expired")
    return FileResponse(path, media_type="application/x-ipynb+json", filename="generated_notebook.ipynb")


@app.post("/api/create-gist/{job_id}")
async def create_gist(job_id: str):
    """Create a GitHub Gist for opening in Colab."""
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(400, "Invalid job ID")
    path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")

    if not os.path.exists(path):
        raise HTTPException(404, f"Notebook not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        notebook_content = f.read()

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise HTTPException(500, "GITHUB_TOKEN not configured in backend")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.github.com/gists",
            json={
                "description": "Paper to Notebook - Generated Notebook",
                "public": True,
                "files": {"notebook.ipynb": {"content": notebook_content}},
            },
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10.0,
        )

    if response.status_code != 201:
        raise HTTPException(500, f"GitHub API error: {response.status_code} - {response.text}")

    gist_data = response.json()
    gist_id = gist_data["id"]
    owner = gist_data["owner"]["login"]
    filename = list(gist_data["files"].keys())[0]
    colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"

    return {"gist_id": gist_id, "gist_url": gist_data["html_url"], "colab_url": colab_url}



@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
