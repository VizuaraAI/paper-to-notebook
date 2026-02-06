"""Gemini API wrapper with native PDF support and retry logic."""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import MAX_RETRIES, RETRY_DELAYS


def load_pdf_as_part(pdf_path: str) -> types.Part:
    """Read a PDF file and return a Gemini API Part for inline data."""
    pdf_bytes = Path(pdf_path).read_bytes()
    return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")


def call_gemini(
    system_prompt: str,
    user_content: list,
    max_tokens: int = 8192,
    model: str = "gemini-2.5-pro",
) -> str:
    """Make a Gemini API call and return the text response."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.7,
        ),
    )
    return response.text


def call_gemini_with_retry(
    system_prompt: str,
    user_content: list,
    max_tokens: int = 8192,
    model: str = "gemini-2.5-pro",
) -> str:
    """Call Gemini API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return call_gemini(system_prompt, user_content, max_tokens, model)

        except Exception as e:
            error_str = str(e).lower()
            # Retry on rate limits and server errors
            if any(keyword in error_str for keyword in ["429", "rate", "500", "503", "overloaded", "unavailable"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  Transient error. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


def parse_llm_json(raw_text: str, step_name: str, model: str) -> dict | list:
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
        repaired = call_gemini_with_retry(
            system_prompt="You are a JSON repair tool. Return only valid JSON.",
            user_content=[repair_prompt],
            max_tokens=max(len(text) // 2, 4096),
            model=model,
        )
        repaired = repaired.strip()
        if repaired.startswith("```"):
            repaired = repaired.split("\n", 1)[1]
        if repaired.endswith("```"):
            repaired = repaired[:-3]
        return json.loads(repaired.strip())
