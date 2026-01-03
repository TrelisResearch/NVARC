"""
Test Gemini 3 Flash with native and OpenAI-compatible APIs.

Tests both API approaches with different thinking levels and token tracking.
Run: uv run python test_gemini_apis.py
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3-flash-preview"
TEST_PROMPT = "What is 2+2? Answer in one word."


def test_native_api(thinking_level="HIGH"):
    """Test native google-genai API with thinking levels (LOW, MEDIUM, HIGH)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )

    start = time.time()
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=TEST_PROMPT)])],
        config=config,
    )
    elapsed = time.time() - start

    usage = response.usage_metadata
    thinking_content = None
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'thought') and part.thought:
                thinking_content = part.text

    return {
        "api": "native",
        "thinking_level": thinking_level,
        "response": response.text,
        "thinking": thinking_content,
        "input_tokens": usage.prompt_token_count if usage else None,
        "output_tokens": usage.candidates_token_count if usage else None,
        "thinking_tokens": usage.thoughts_token_count if usage else None,
        "elapsed_seconds": round(elapsed, 2),
    }


def test_openai_compatible_api(reasoning_effort="high"):
    """Test OpenAI-compatible API. reasoning_effort maps to thinking_level."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        extra_body={"reasoning_effort": reasoning_effort},
    )
    elapsed = time.time() - start

    usage = response.usage
    # Thinking tokens = total - prompt - completion (not directly exposed)
    thinking_tokens = (usage.total_tokens - usage.prompt_tokens - usage.completion_tokens) if usage else None

    return {
        "api": "openai-compatible",
        "reasoning_effort": reasoning_effort,
        "response": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "thinking_tokens": thinking_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    print("=" * 60)
    print(f"Testing Gemini 3 Flash APIs ({MODEL})")
    print("=" * 60)

    # Test native API with different thinking levels
    for level in ["LOW", "MEDIUM", "HIGH"]:
        print(f"\n--- Native API, thinking_level={level} ---")
        try:
            r = test_native_api(thinking_level=level)
            print(f"Response: {r['response']}")
            print(f"Tokens: {r['input_tokens']} in / {r['output_tokens']} out / {r['thinking_tokens']} thinking")
            print(f"Time: {r['elapsed_seconds']}s")
        except Exception as e:
            print(f"Error: {e}")

    # Test OpenAI-compatible API with different reasoning_effort levels
    for effort in ["low", "high"]:
        print(f"\n--- OpenAI-compatible API, reasoning_effort={effort} ---")
        try:
            r = test_openai_compatible_api(reasoning_effort=effort)
            print(f"Response: {r['response']}")
            print(f"Tokens: {r['input_tokens']} in / {r['output_tokens']} out / {r['thinking_tokens']} thinking")
            print(f"Time: {r['elapsed_seconds']}s")
        except Exception as e:
            print(f"Error: {e}")
