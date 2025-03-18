import subprocess
import requests
from config import LLM_MODE, LLAMA_MODEL, TOGETHERAI_API_KEY
from src.cache_manager import get_cached_response, cache_response


def summarize_text(text, length="medium"):
    """Summarize text using Llama 3 via Ollama or TogetherAI."""

    # cache_key = f"{text}-{length}"

    cached_summary = get_cached_response(text + length)
    if cached_summary:
        return cached_summary

    if len(text.split()) <= 30:  # If text is short, skip summarization
        return text

    prompt = f"Summarize the following text in a {length} length:\n\n{text}"

    if LLM_MODE == "ollama":
        command = f'ollama run {LLAMA_MODEL} "{prompt}"'
        response = subprocess.run(command, shell=True, capture_output=True, text=True)
        summary = response.stdout.strip()

    elif LLM_MODE == "togetherai":
        headers = {"Authorization": f"Bearer {TOGETHERAI_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300
        }

        try:
            response = requests.post("https://api.together.xyz/v1/chat/completions", json=data, headers=headers)
            response_json = response.json()

            if "choices" in response_json and len(response_json["choices"]) > 0:
                summary = response_json["choices"][0]["message"]["content"]
            else:
                print(f"❌ Unexpected API response: {response_json}")
                summary = "Error: API response format invalid."

        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling TogetherAI API: {e}")
            summary = "Error: Unable to reach TogetherAI API."

    else:
        raise ValueError("❌ Invalid LLM_MODE. Use 'ollama' or 'togetherai'.")

    # cache_response(text, summary)
    cache_response(text + length, summary)
    return summary
