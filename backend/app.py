#!/usr/bin/env python3
"""
Paper to Notebook - Backend API
Single-file FastAPI application for deploying on Railway.
Converts research paper PDFs into executable Jupyter notebooks using Gemini LLM.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Literal

import httpx
import nbformat
import pypdf
import re
from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from google import genai
from google.genai import types
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Provider types
LLM_PROVIDER = Literal["gemini", "ollama"]

# Default Gemini model
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_PROVIDER: LLM_PROVIDER = "gemini"

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Ollama available models
OLLAMA_MODELS = [
    "gpt-oss:120b-cloud",
    "mistral:7b",
    "phi4:14b",
    "qwen2.5:1.5b",
    "deepseek-r1:14b",
    "mistral-small:22b",
    "mistral-small:24b",
    "gpt-oss:20b",
]

# Token limits per pipeline step
MAX_TOKENS_ANALYSIS = 8192
MAX_TOKENS_DESIGN = 8192
MAX_TOKENS_GENERATE = 65536
MAX_TOKENS_VALIDATE = 65536

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]  # seconds

# PDF constraints
MAX_PDF_SIZE_MB = 30

# Upload configuration
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", str(MAX_PDF_SIZE_MB)))



# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = (
    "You are an expert research engineer and educator who faithfully implements "
    "academic papers as runnable, educational Python code. You use real ML components "
    "(PyTorch, Transformer layers, actual training loops) at a reduced scale that "
    "runs on CPU. You prioritize faithful replication of the paper's architecture "
    "and algorithms while making the code deeply educational with clear explanations, "
    "verbose logging, and insightful visualizations."
)

ANALYSIS_PROMPT = """\
Read this research paper carefully and extract a thorough structured analysis.

Return a JSON object with EXACTLY these fields:

{
  "title": "Full paper title",
  "authors": "Author list as a single string",
  "abstract_summary": "2-3 sentence plain English summary of the paper",
  "problem_statement": "What problem does the paper solve? (2-3 sentences, no jargon)",
  "key_insight": "The core idea or innovation in one sentence",
  "algorithms": [
    {
      "name": "Algorithm name (e.g., 'GRPO', 'DPO', 'LLaDA Pre-training', etc.)",
      "description": "What this algorithm does in plain English",
      "inputs": ["list of inputs with types and shapes where applicable"],
      "outputs": ["list of outputs with types and shapes where applicable"],
      "steps": ["ordered list of DETAILED algorithmic steps — include math operations, loss functions, gradient updates"],
      "is_core": true,
      "equations": ["key equations used in this algorithm, in LaTeX or descriptive form"],
      "architecture_details": "Describe the neural network architecture used (layers, dimensions, attention type, etc.)"
    }
  ],
  "baselines": [
    {
      "name": "Baseline method name",
      "description": "Detailed description of how it works, including its loss function and training procedure"
    }
  ],
  "evaluation_metrics": ["list of metrics used to evaluate, with formulas if available"],
  "key_equations": ["ALL important equations from the paper described precisely"],
  "model_architecture": {
    "type": "Transformer/CNN/RNN/etc.",
    "key_layers": ["list of layer types used"],
    "dimensions": "hidden dim, num heads, num layers mentioned in paper",
    "special_features": "any non-standard architectural choices"
  },
  "dataset": {
    "name": "Dataset name if mentioned",
    "description": "Brief description",
    "preprocessing": "Any special preprocessing steps"
  },
  "research_field": "Primary research field in 2-4 words (e.g., 'Natural Language Processing', 'Computer Vision', 'Reinforcement Learning', 'Graph Neural Networks')",
  "key_contributions": ["Contribution in 4-7 words", "Contribution in 4-7 words", "Contribution in 4-7 words"]
}

Be exhaustive. Extract every algorithmic detail, equation, and architectural choice.
"""

DESIGN_PROMPT_TEMPLATE = """\
You are given the analysis of a research paper (below). Your job is to design a **toy implementation plan** for a Jupyter notebook that demonstrates the paper's core ideas using real PyTorch code at a small, CPU-runnable scale.

**Paper Analysis:**
```json
{analysis_json}
```

Return a JSON object with:

{{
  "notebook_title": "A clear, descriptive title for the notebook",
  "model_architecture": {{
    "type": "Transformer/CNN/RNN/etc.",
    "embed_dim": 64,
    "num_layers": 2,
    "num_heads": 4,
    "vocab_size": 1000,
    "max_seq_len": 32,
    "other_params": {{}}
  }},
  "synthetic_data": {{
    "description": "What kind of synthetic/toy data to generate",
    "size": "e.g., 500 training samples, 100 test samples",
    "generation_method": "How to create it (random, simple patterns, etc.)"
  }},
  "training_config": {{
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss or custom"
  }},
  "mock_models": [
    {{
      "name": "BaselineModel",
      "purpose": "What baseline this represents",
      "architecture_summary": "Brief description of layers"
    }},
    {{
      "name": "PaperModel",
      "purpose": "The paper's proposed method",
      "architecture_summary": "Brief description"
    }}
  ],
  "visualizations": [
    {{
      "type": "training curves",
      "description": "Loss over time for both models"
    }},
    {{
      "type": "metric comparison",
      "description": "Bar chart comparing baseline vs paper method"
    }}
  ],
  "implementation_notes": "Any important considerations or simplifications"
}}

Make it realistic but small-scale. Focus on educational clarity.
"""

GENERATE_PROMPT_TEMPLATE = """\
You have analyzed a research paper and designed a toy implementation plan. Now generate the **complete Jupyter notebook** as a list of cells.

**Paper Analysis:**
```json
{analysis_json}
```

**Design Plan:**
```json
{design_json}
```

Return a JSON array of notebook cells following this **exact 11-section structure**:

1. **Title & Paper Overview** (markdown) — Paper title, authors, one-paragraph summary
2. **Problem Intuition** (markdown) — Explain the problem in simple terms with an analogy
3. **Imports & Setup** (code) — All imports, set random seeds, device setup
4. **Dataset & Tokenization** (code + markdown) — Generate synthetic data, show samples
5. **Model Architecture** (code + markdown) — Define the PyTorch model class(es)
6. **Loss Function & Training Utilities** (code) — Loss function, training loop helper
7. **Baseline Implementation** (code + markdown) — Simple baseline model
8. **Paper's Main Algorithm — Training** (code + markdown) — Implement paper's method
9. **Inference / Generation** (code + markdown) — Run inference, show predictions
10. **Full Experiment & Evaluation** (code) — Train both models, compute metrics
11. **Visualizations** (code) — Plot training curves, comparison charts
12. **Summary & Next Steps** (markdown) — What we learned, ideas for extension

Each cell must be:
```json
{{
  "cell_type": "code" | "markdown",
  "source": "the full cell content as a string"
}}
```

**CRITICAL REQUIREMENTS:**
- Use REAL PyTorch (torch.nn.Module, torch.optim, actual training loops)
- NO placeholders like "# TODO" or "pass" — write complete working code
- Include plenty of print statements for educational insight
- Add comments explaining key lines
- Make it runnable on CPU with small data
- Follow the 11-section structure exactly

Return ONLY the JSON array of cells, no extra text.
"""

VALIDATE_PROMPT_TEMPLATE = """\
You are given a list of Jupyter notebook cells. Your job is to **validate and repair** them.

**Cells:**
```json
{cells_json}
```

Check for:
1. **Undefined variables** — every variable must be defined before use
2. **Missing imports** — all libraries must be imported
3. **Syntax errors** — valid Python syntax
4. **Logical flow** — cells execute in order without errors
5. **No placeholders** — no "# TODO", "pass", or "..." in critical code
6. **Complete implementations** — all functions fully implemented

Return the **corrected cells** as a JSON array with the same structure:
```json
[
  {{
    "cell_type": "code" | "markdown",
    "source": "corrected content"
  }},
  ...
]
```

Fix any issues you find. If cells are already correct, return them unchanged.
Return ONLY the JSON array, no extra text.
"""

# ============================================================================
# LLM UTILITIES
# ============================================================================

def _get_api_key(api_key: str | None = None) -> str:
    """Get API key from parameter, environment, or fallback."""
    return api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") 


def call_ollama(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 8192,
    model: str = "mistral:7b",
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Make an Ollama API call and return the text response."""
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": user_content,
        "system": system_prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "num_ctx": 16384,  # Increase context window for large papers
        },
    }
    
    try:
        response = httpx.post(url, json=payload, timeout=300.0)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to call Ollama: {str(e)}")


def call_ollama_with_retry(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 8192,
    model: str = "mistral:7b",
    base_url: str = OLLAMA_BASE_URL,
) -> str:
    """Call Ollama API with retry logic for transient errors."""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return call_ollama(system_prompt, user_content, max_tokens, model, base_url)
        except Exception as e:
            error_str = str(e).lower()
            last_error = e
            
            # Retry for transient errors
            if any(keyword in error_str for keyword in ["connection", "timeout", "refused"]):
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise
    
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def call_llm(
    system_prompt: str,
    user_content: list | str,
    max_tokens: int = 8192,
    model: str = DEFAULT_MODEL,
    provider: LLM_PROVIDER = DEFAULT_PROVIDER,
    api_key: str | None = None,
    base_url: str = OLLAMA_BASE_URL,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Unified LLM call function that works with both Gemini and Ollama."""
    if provider == "ollama":
        # Convert user_content to string for Ollama
        if isinstance(user_content, list):
            content_str = ""
            for item in user_content:
                if isinstance(item, str):
                    content_str += item + "\n"
        else:
            content_str = user_content
        return call_ollama_with_retry(system_prompt, content_str, max_tokens, model, base_url)
    else:  # gemini
        # For Gemini, keep user_content as list
        if isinstance(user_content, str):
            user_content = [user_content]
        return call_gemini_with_retry(system_prompt, user_content, max_tokens, model, api_key, on_thinking)


def call_gemini(
    system_prompt: str,
    user_content: list,
    max_tokens: int = 8192,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Make a Gemini API call and return the text response."""
    client = genai.Client(api_key=_get_api_key(api_key))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.7,
    )

    if on_thinking:
        thinking_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.7,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
        full_text = ""
        for chunk in client.models.generate_content_stream(
            model=model, contents=user_content, config=thinking_config
        ):
            try:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, 'thought', False):
                            if part.text:
                                on_thinking(part.text)
                        else:
                            if part.text:
                                full_text += part.text
            except (AttributeError, IndexError):
                if hasattr(chunk, 'text') and chunk.text:
                    full_text += chunk.text
        return full_text
    else:
        response = client.models.generate_content(
            model=model, contents=user_content, config=config
        )
        return response.text


def call_gemini_with_retry(
    system_prompt: str,
    user_content: list,
    max_tokens: int = 8192,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> str:
    """Call Gemini API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_gemini(system_prompt, user_content, max_tokens, model, api_key, on_thinking)

        except Exception as e:
            error_str = str(e).lower()

            # Check for invalid API key error
            if any(keyword in error_str for keyword in ["api key not valid", "api_key_invalid", "invalid_argument"]):
                raise ValueError("Invalid API key. Please check your Gemini API key and try again.")

            # Retry for transient errors
            if any(keyword in error_str for keyword in ["429", "rate", "500", "503", "overloaded", "unavailable"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def parse_llm_json(raw_text: str, step_name: str, model: str, provider: LLM_PROVIDER = DEFAULT_PROVIDER, api_key: str | None = None, base_url: str = OLLAMA_BASE_URL) -> dict | list:
    """Parse JSON from LLM response, with cleanup and one repair attempt."""
    text = raw_text.strip()

    # Strip markdown code fences if present
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
        repaired = call_llm(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=repair_prompt,
            max_tokens=max(len(text) // 2, 4096),
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        repaired = repaired.strip()
        if repaired.startswith("```"):
            repaired = repaired.split("\n", 1)[1]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        return json.loads(repaired.strip())


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes for non-multimodal LLMs."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


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
    """Convert notebook to bytes."""
    buffer = io.StringIO()
    nbformat.write(nb, buffer)
    return buffer.getvalue().encode("utf-8")


def cells_to_bytes(cells: list) -> bytes:
    """Convert cells to notebook bytes."""
    nb = build_notebook(cells)
    return nb_to_bytes(nb)

# ============================================================================
# PIPELINE
# ============================================================================

ProgressCallback = Callable[[int, str, str, Optional[dict]], None]
ThinkingCallback = Callable[[str], None]


def run_pipeline(
    pdf_bytes: bytes,
    model: str = DEFAULT_MODEL,
    provider: LLM_PROVIDER = DEFAULT_PROVIDER,
    on_progress: Optional[ProgressCallback] = None,
    api_key: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    on_thinking: Optional[ThinkingCallback] = None,
) -> bytes:
    """Run the full pipeline on PDF bytes, returning .ipynb bytes."""

    def _notify(step: int, name: str, detail: str = "", extra: Optional[dict] = None):
        if on_progress:
            on_progress(step, name, detail, extra)

    # For Gemini, use PDF binary data. For Ollama, we'll use text-based analysis
    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf") if provider == "gemini" else None

    # Step 1: Paper Analysis
    _notify(1, "Analyzing paper", "Reading PDF and extracting structure...")
    
    
    if provider == "gemini":
        analysis_raw = call_gemini_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_content=[pdf_part, ANALYSIS_PROMPT],
            max_tokens=MAX_TOKENS_ANALYSIS,
            model=model,
            api_key=api_key,
            on_thinking=on_thinking,
        )
    else:  # ollama
        # Extract text from PDF first
        _notify(1, "Analyzing paper", "Extracting text from PDF for local analysis...")
        paper_text = extract_text_from_pdf(pdf_bytes)
        
        # Combine prompt and paper text
        combined_prompt = f"{ANALYSIS_PROMPT}\n\n==== PAPER CONTENT ====\n{paper_text[:100000]}"  # Truncate if too huge
        
        analysis_raw = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_content=combined_prompt,
            max_tokens=MAX_TOKENS_ANALYSIS,
            model=model,
            provider=provider,
            base_url=base_url,
        )
    
    analysis = parse_llm_json(analysis_raw, "paper_analysis", model, provider=provider, api_key=api_key, base_url=base_url)
    title = analysis.get("title", "Unknown Paper")
    num_algos = len(analysis.get("algorithms", []))
    # Clean up metrics: strip formula parts (anything after = or ()
    raw_metrics = analysis.get("evaluation_metrics", [])
    def _clean_metric(m: str) -> str:
        m = re.split(r'\s*[=(]', m)[0].strip().rstrip(',')
        return m
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
    })

    # Step 2: Design Plan
    _notify(2, "Designing implementation", "Planning model architecture and training...")
    design_prompt = DESIGN_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2)
    )
    if provider == "gemini":
        design_raw = call_gemini_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_content=[pdf_part, design_prompt],
            max_tokens=MAX_TOKENS_DESIGN,
            model=model,
            api_key=api_key,
            on_thinking=on_thinking,
        )
    else:  # ollama
        design_raw = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_content=design_prompt,
            max_tokens=MAX_TOKENS_DESIGN,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
    
    design = parse_llm_json(design_raw, "toy_design", model, provider=provider, api_key=api_key, base_url=base_url)
    arch = design.get("model_architecture", {})
    _notify(2, "Designing implementation", "Architecture designed", {
        "type": "design",
        "notebook_title": design.get("notebook_title", ""),
        "model_type": arch.get("type", ""),
        "embed_dim": arch.get("embed_dim", ""),
        "num_layers": arch.get("num_layers", ""),
        "num_heads": arch.get("num_heads", ""),
    })

    # Step 3: Generate Notebook Cells
    _notify(3, "Generating notebook", "Writing PyTorch code and explanations...")
    generate_prompt = GENERATE_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2),
        design_json=json.dumps(design, indent=2),
    )
    if provider == "gemini":
        cells_raw = call_gemini_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_content=[pdf_part, generate_prompt],
            max_tokens=MAX_TOKENS_GENERATE,
            model=model,
            api_key=api_key,
            on_thinking=on_thinking,
        )
    else:  # ollama
        cells_raw = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_content=generate_prompt,
            max_tokens=MAX_TOKENS_GENERATE,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
    
    cells = parse_llm_json(cells_raw, "generate_cells", model, provider=provider, api_key=api_key, base_url=base_url)
    num_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    previews = []
    for c in cells:
        previews.append({
            "type": c.get("cell_type", "code"),
            "preview": c.get("source", "")[:300],
        })

    # Build draft notebook bytes and send as draft_ready
    draft_bytes = cells_to_bytes(cells)
    _notify(3, "Generating notebook", f"Generated {num_cells} cells ({code_cells} code)", {
        "type": "cells_generated",
        "num_cells": num_cells,
        "code_cells": code_cells,
        "previews": previews,
        "draft_bytes": draft_bytes,
    })

    # Step 4: Validate & Repair (LLM review)
    _notify(4, "Validating code", "LLM reviewing for errors...")
    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(
        cells_json=json.dumps(cells, indent=2)
    )
    if provider == "gemini":
        validated_raw = call_gemini_with_retry(
            system_prompt=SYSTEM_PROMPT,
            user_content=[validate_prompt],
            max_tokens=MAX_TOKENS_VALIDATE,
            model=model,
            api_key=api_key,
            on_thinking=on_thinking,
        )
    else:  # ollama
        validated_raw = call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_content=validate_prompt,
            max_tokens=MAX_TOKENS_VALIDATE,
            model=model,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
    
    validated_cells = parse_llm_json(validated_raw, "validate", model, provider=provider, api_key=api_key, base_url=base_url)
    _notify(4, "Validating code", "Validation complete")

    # Build and return validated notebook
    nb = build_notebook(validated_cells)
    return nb_to_bytes(nb)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Paper to Notebook API",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for generated notebooks
TEMP_DIR = tempfile.mkdtemp(prefix="paper2nb_")

# Concurrency limiter
_generation_semaphore = asyncio.Semaphore(3)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Paper to Notebook API",
        "version": "2.0",
        "endpoints": {
            "generate": "/api/generate",
            "download": "/api/download/{job_id}",
            "health": "/health",
            "models": "/api/models",
        }
    }


@app.get("/api/models")
async def get_models():
    """Get available models by provider."""
    return {
        "providers": ["gemini", "ollama"],
        "models": {
            "gemini": [DEFAULT_MODEL],
            "ollama": OLLAMA_MODELS,
        },
        "ollama_base_url": OLLAMA_BASE_URL,
    }


@app.post("/api/generate-from-arxiv")
async def generate_from_arxiv(
    request: Request,
    arxiv_url: str = Form(...),
    api_key: str = Form(...),
    provider: str = Form(DEFAULT_PROVIDER),
    model: str = Form(DEFAULT_MODEL),
):
    """Generate notebook from arXiv URL."""

    api_key = api_key.strip()
    if not api_key and provider == "gemini":
        raise HTTPException(400, "Gemini API key is required for Gemini provider")

    # Validate provider
    if provider not in ["gemini", "ollama"]:
        raise HTTPException(400, f"Invalid provider. Must be 'gemini' or 'ollama'")

    # Validate model
    if provider == "gemini" and model != DEFAULT_MODEL:
        # For now, only support the default Gemini model
        pass
    elif provider == "ollama" and model not in OLLAMA_MODELS:
        raise HTTPException(400, f"Invalid Ollama model. Must be one of: {', '.join(OLLAMA_MODELS)}")

    # Extract arXiv paper ID from URL
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)', arxiv_url)
    if not match:
        raise HTTPException(400, "Invalid arXiv URL. Expected format: https://arxiv.org/abs/XXXX.XXXXX")

    paper_id = match.group(1)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    # Download PDF from arXiv
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            print(f"Downloading PDF from arXiv: {pdf_url}")
            response = await client.get(pdf_url, timeout=30.0)

            if response.status_code != 200:
                raise HTTPException(500, f"Failed to download PDF from arXiv: {response.status_code}")

            pdf_bytes = response.content

            # Check file size
            size_mb = len(pdf_bytes) / (1024 * 1024)
            if size_mb > MAX_UPLOAD_MB:
                raise HTTPException(413, f"PDF too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")

            print(f"Downloaded PDF: {size_mb:.1f} MB")

    except httpx.HTTPError as e:
        raise HTTPException(500, f"Failed to download PDF from arXiv: {str(e)}")

    # Continue with same pipeline as file upload
    job_id = uuid.uuid4().hex[:12]
    draft_id = job_id + "_draft"

    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_progress(step: int, name: str, detail: str, extra: dict = None):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("progress", step, name, detail, extra)),
                loop,
            )

        def on_thinking(text: str):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("thinking", text)),
                loop,
            )

        async def run_in_thread():
            async with _generation_semaphore:
                return await loop.run_in_executor(
                    None,
                    lambda: run_pipeline(
                        pdf_bytes,
                        model=model,
                        provider=provider,
                        on_progress=on_progress,
                        api_key=api_key,
                        base_url=OLLAMA_BASE_URL,
                        on_thinking=on_thinking,
                    ),
                )

        task = asyncio.create_task(run_in_thread())

        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)

                if event[0] == "thinking":
                    data = json.dumps({"text": event[1]})
                    yield f"event: thinking\ndata: {data}\n\n"
                    continue

                _, step, name, detail, extra = event

                if extra and "draft_bytes" in extra:
                    draft_bytes = extra.pop("draft_bytes")
                    draft_path = os.path.join(TEMP_DIR, f"{draft_id}.ipynb")
                    with open(draft_path, "wb") as f:
                        f.write(draft_bytes)
                    data = {"step": step, "name": name, "detail": detail, "extra": extra}
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                    draft_data = json.dumps({"job_id": draft_id, "size_kb": len(draft_bytes) // 1024})
                    yield f"event: draft_ready\ndata: {draft_data}\n\n"
                else:
                    data = {"step": step, "name": name, "detail": detail}
                    if extra:
                        data["extra"] = extra
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

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

        try:
            notebook_bytes = task.result()
            output_path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")
            with open(output_path, "wb") as f:
                f.write(notebook_bytes)
            data = json.dumps({"job_id": job_id, "size_kb": len(notebook_bytes) // 1024})
            yield f"event: complete\ndata: {data}\n\n"
        except Exception as e:
            data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate")
async def generate(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    provider: str = Form(DEFAULT_PROVIDER),
    model: str = Form(DEFAULT_MODEL),
):
    """Generate notebook from PDF with streaming progress."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    api_key = api_key.strip()
    if not api_key and provider == "gemini":
        raise HTTPException(400, "Gemini API key is required for Gemini provider")

    # Validate provider
    if provider not in ["gemini", "ollama"]:
        raise HTTPException(400, f"Invalid provider. Must be 'gemini' or 'ollama'")

    # Validate model
    if provider == "gemini" and model != DEFAULT_MODEL:
        # For now, only support the default Gemini model
        pass
    elif provider == "ollama" and model not in OLLAMA_MODELS:
        raise HTTPException(400, f"Invalid Ollama model. Must be one of: {', '.join(OLLAMA_MODELS)}")

    pdf_bytes = await file.read()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"PDF too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.")

    job_id = uuid.uuid4().hex[:12]
    draft_id = job_id + "_draft"

    async def event_stream():
        loop = asyncio.get_event_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_progress(step: int, name: str, detail: str, extra: dict = None):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("progress", step, name, detail, extra)),
                loop,
            )

        def on_thinking(text: str):
            asyncio.run_coroutine_threadsafe(
                progress_queue.put(("thinking", text)),
                loop,
            )

        async def run_in_thread():
            async with _generation_semaphore:
                return await loop.run_in_executor(
                    None,
                    lambda: run_pipeline(
                        pdf_bytes,
                        model=model,
                        provider=provider,
                        on_progress=on_progress,
                        api_key=api_key,
                        base_url=OLLAMA_BASE_URL,
                        on_thinking=on_thinking,
                    ),
                )

        task = asyncio.create_task(run_in_thread())

        while not task.done():
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)

                if event[0] == "thinking":
                    data = json.dumps({"text": event[1]})
                    yield f"event: thinking\ndata: {data}\n\n"
                    continue

                _, step, name, detail, extra = event

                # Check if this progress event carries draft notebook bytes
                if extra and "draft_bytes" in extra:
                    draft_bytes = extra.pop("draft_bytes")
                    # Save draft to disk
                    draft_path = os.path.join(TEMP_DIR, f"{draft_id}.ipynb")
                    with open(draft_path, "wb") as f:
                        f.write(draft_bytes)
                    # Send progress event (without the bytes)
                    data = {"step": step, "name": name, "detail": detail, "extra": extra}
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                    # Send draft_ready event
                    draft_data = json.dumps({"job_id": draft_id, "size_kb": len(draft_bytes) // 1024})
                    yield f"event: draft_ready\ndata: {draft_data}\n\n"
                else:
                    data = {"step": step, "name": name, "detail": detail}
                    if extra:
                        data["extra"] = extra
                    yield f"event: progress\ndata: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

        # Drain remaining
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

        try:
            notebook_bytes = task.result()
            output_path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")
            with open(output_path, "wb") as f:
                f.write(notebook_bytes)
            data = json.dumps({"job_id": job_id, "size_kb": len(notebook_bytes) // 1024})
            yield f"event: complete\ndata: {data}\n\n"
        except Exception as e:
            data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {data}\n\n"

    return StreamingResponse(
        event_stream(),
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
    return FileResponse(
        path,
        media_type="application/x-ipynb+json",
        filename="generated_notebook.ipynb",
    )


@app.post("/api/create-gist/{job_id}")
async def create_gist(job_id: str):
    """Create a GitHub Gist for opening in Colab."""
    try:
        import httpx
    except ImportError:
        raise HTTPException(500, "httpx not installed. Run: pip install httpx")

    # Handle test notebook
    if job_id == "test":
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "generated_notebook (1).ipynb")
        print(f"Test notebook path: {path}")
        print(f"Test notebook exists: {os.path.exists(path)}")
    else:
        if not job_id.replace("_", "").isalnum():
            raise HTTPException(400, "Invalid job ID")
        path = os.path.join(TEMP_DIR, f"{job_id}.ipynb")

    if not os.path.exists(path):
        print(f"ERROR: Notebook not found at: {path}")
        raise HTTPException(404, f"Notebook not found at: {path}")

    # Read notebook content
    try:
        with open(path, "r", encoding="utf-8") as f:
            notebook_content = f.read()
    except Exception as e:
        print(f"Error reading notebook: {e}")
        raise HTTPException(500, f"Failed to read notebook: {str(e)}")

    # Get GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise HTTPException(500, "GITHUB_TOKEN not configured in backend")

    # Create GitHub Gist
    try:
        async with httpx.AsyncClient() as client:
            print("Creating GitHub Gist...")
            response = await client.post(
                "https://api.github.com/gists",
                json={
                    "description": "Paper to Notebook - Generated Notebook",
                    "public": True,
                    "files": {
                        "notebook.ipynb": {
                            "content": notebook_content
                        }
                    }
                },
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {github_token}",
                    "X-GitHub-Api-Version": "2022-11-28"
                },
                timeout=10.0
            )

            print(f"GitHub API response status: {response.status_code}")

            if response.status_code != 201:
                error_detail = response.text
                print(f"GitHub API error: {error_detail}")
                raise HTTPException(500, f"GitHub API error: {response.status_code} - {error_detail}")

            gist_data = response.json()
            gist_id = gist_data["id"]
            owner = gist_data["owner"]["login"]
            # Get the first filename from the files dict
            filename = list(gist_data["files"].keys())[0]

            # Construct proper Colab URL
            colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"

            print(f"Gist created successfully: {gist_id}")
            print(f"Colab URL: {colab_url}")

            return {
                "gist_id": gist_id,
                "gist_url": gist_data["html_url"],
                "colab_url": colab_url
            }

    except httpx.HTTPError as e:
        print(f"HTTP error creating Gist: {e}")
        raise HTTPException(500, f"Network error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error creating Gist: {e}")
        raise HTTPException(500, f"Failed to create Gist: {str(e)}")


@app.get("/api/test-notebook")
async def test_notebook():
    """Serve the test notebook for Colab testing."""
    test_path = os.path.join(os.path.dirname(__file__), "generated_notebook (1).ipynb")
    if not os.path.exists(test_path):
        raise HTTPException(404, "Test notebook not found")
    return FileResponse(
        test_path,
        media_type="application/x-ipynb+json",
        filename="test_notebook.ipynb",
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
