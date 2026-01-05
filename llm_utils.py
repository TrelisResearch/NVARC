"""
Shared LLM utilities for ARC task validation.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gemini-3-flash-preview"
PROMPTS_DIR = Path(__file__).parent / "SDG" / "prompts"
TASKS_FILE = Path(__file__).parent / "arc_agi2_training_only" / "arc-agi_training_challenges.json"
SOLUTIONS_FILE = Path(__file__).parent / "arc_agi2_training_only" / "arc-agi_training_solutions.json"

# Pricing per 1M tokens
INPUT_PRICE = 0.50
OUTPUT_PRICE = 3.00


def get_client():
    return OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )


def call_gemini(client: OpenAI, prompt: str, reasoning_effort: str = "high") -> dict:
    """Call Gemini 3 Flash and return response with token usage."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning_effort": reasoning_effort},
    )
    usage = response.usage
    thinking_tokens = (usage.total_tokens - usage.prompt_tokens - usage.completion_tokens) if usage else 0
    return {
        "content": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "thinking_tokens": thinking_tokens,
    }
