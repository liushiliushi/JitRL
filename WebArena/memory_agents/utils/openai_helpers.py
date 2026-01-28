import time
from typing import Mapping, Optional, Any
import openai
from openai import OpenAI
import tiktoken
import os
import json
import logging
from dotenv import load_dotenv

# Force reload from .env file, overriding system environment variables
load_dotenv(override=True)

# Suppress httpx INFO logs (like HTTP request logs)
logging.getLogger("httpx").setLevel(logging.WARNING)
encoding = tiktoken.get_encoding("cl100k_base")

# Global cache for local models (to avoid reloading on each call)
_LOCAL_MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_path": None,
    "backend": None,  # "vllm" or "transformers"
}


def _is_local_model(model_name: str) -> bool:
    """Check if the model is a local model path (not an API model)"""
    # Local model paths typically start with "/" or contain common path patterns
    if model_name.startswith("/"):
        return True
    if model_name.startswith("./") or model_name.startswith("../"):
        return True
    # Check for common local model directory patterns
    if any(pattern in model_name for pattern in ["/ndata/", "/home/", "/data/", "/models/"]):
        return True
    return False


def _init_local_model(model_path: str, backend: str = "auto"):
    """
    Initialize a local model for inference.

    Args:
        model_path: Path to the local model checkpoint
        backend: "vllm", "transformers", or "auto" (try vllm first, fallback to transformers)

    Returns:
        Tuple of (model, tokenizer, backend_used)
    """
    global _LOCAL_MODEL_CACHE

    # Return cached model if same path
    if _LOCAL_MODEL_CACHE["model_path"] == model_path and _LOCAL_MODEL_CACHE["model"] is not None:
        print(f"[Local Model] Using cached model from {model_path}")
        return _LOCAL_MODEL_CACHE["model"], _LOCAL_MODEL_CACHE["tokenizer"], _LOCAL_MODEL_CACHE["backend"]

    print(f"[Local Model] Loading model from {model_path}...")

    # Determine backend
    if backend == "auto":
        try:
            from vllm import LLM
            backend = "vllm"
            print("[Local Model] Using vLLM backend")
        except ImportError:
            backend = "transformers"
            print("[Local Model] vLLM not available, using transformers backend")

    if backend == "vllm":
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Load model with vLLM
            model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,  # Adjust based on available GPUs
                gpu_memory_utilization=0.9,
            )

            _LOCAL_MODEL_CACHE["model"] = model
            _LOCAL_MODEL_CACHE["tokenizer"] = tokenizer
            _LOCAL_MODEL_CACHE["model_path"] = model_path
            _LOCAL_MODEL_CACHE["backend"] = "vllm"

            print(f"[Local Model] Successfully loaded with vLLM: {model_path}")
            return model, tokenizer, "vllm"

        except Exception as e:
            print(f"[Local Model] vLLM loading failed: {e}, falling back to transformers")
            backend = "transformers"

    if backend == "transformers":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        _LOCAL_MODEL_CACHE["model"] = model
        _LOCAL_MODEL_CACHE["tokenizer"] = tokenizer
        _LOCAL_MODEL_CACHE["model_path"] = model_path
        _LOCAL_MODEL_CACHE["backend"] = "transformers"

        print(f"[Local Model] Successfully loaded with transformers: {model_path}")
        return model, tokenizer, "transformers"

    raise ValueError(f"Unknown backend: {backend}")


def _local_model_chat_completion(
    model_path: str,
    sys_prompt: str,
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    top_logprobs: int = 0,
    **kwargs
) -> Mapping:
    """
    Generate chat completion using a local model.

    Returns a response object that mimics OpenAI's API response format.
    """
    model, tokenizer, backend = _init_local_model(model_path)

    # Build messages in chat format
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback for tokenizers without chat template
        input_text = f"System: {sys_prompt}\n\nUser: {prompt}\n\nAssistant:"

    if backend == "vllm":
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.01,
            max_tokens=max_tokens,
            logprobs=top_logprobs if top_logprobs > 0 else None,
        )

        outputs = model.generate([input_text], sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text

        # Build response object mimicking OpenAI format
        response = _build_openai_like_response(
            generated_text=generated_text,
            model_name=model_path,
            logprobs_data=output.outputs[0].logprobs if top_logprobs > 0 else None,
            top_logprobs=top_logprobs,
        )

    else:  # transformers
        import torch

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=top_logprobs > 0,
                return_dict_in_generate=True,
            )

        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Build logprobs if requested
        logprobs_data = None
        if top_logprobs > 0 and hasattr(outputs, "scores") and outputs.scores:
            logprobs_data = _extract_transformers_logprobs(
                outputs.scores,
                generated_ids,
                tokenizer,
                top_logprobs
            )

        response = _build_openai_like_response(
            generated_text=generated_text,
            model_name=model_path,
            logprobs_data=logprobs_data,
            top_logprobs=top_logprobs,
        )

    return response


def _extract_transformers_logprobs(scores, generated_ids, tokenizer, top_k):
    """Extract top logprobs from transformers generation scores."""
    import torch
    import torch.nn.functional as F

    logprobs_list = []

    for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
        # Get log probabilities
        log_probs = F.log_softmax(score[0], dim=-1)

        # Get top-k tokens and their logprobs
        top_values, top_indices = torch.topk(log_probs, top_k)

        top_logprobs = []
        for value, idx in zip(top_values.tolist(), top_indices.tolist()):
            token_str = tokenizer.decode([idx])
            top_logprobs.append({
                "token": token_str,
                "logprob": value,
                "bytes": None,
            })

        # Current token info
        token_logprob = log_probs[token_id].item()
        token_str = tokenizer.decode([token_id.item()])

        logprobs_list.append({
            "token": token_str,
            "logprob": token_logprob,
            "bytes": None,
            "top_logprobs": top_logprobs,
        })

    return logprobs_list


def _build_openai_like_response(generated_text: str, model_name: str, logprobs_data=None, top_logprobs=0):
    """Build a response object that mimics OpenAI's chat completion response."""

    # Create a simple namespace object to mimic OpenAI response
    class Choice:
        def __init__(self):
            self.message = type('Message', (), {'content': generated_text, 'role': 'assistant'})()
            self.finish_reason = "stop"
            self.index = 0
            self.logprobs = None

            if logprobs_data and top_logprobs > 0:
                self.logprobs = type('Logprobs', (), {'content': logprobs_data})()

    class Response:
        def __init__(self):
            self.choices = [Choice()]
            self.model = model_name
            self.id = f"local-{int(time.time())}"
            self.created = int(time.time())
            self.object = "chat.completion"
            self.usage = type('Usage', (), {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            })()

    return Response()

def _is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI model (should use OpenAI API directly)"""
    return model_name.startswith(("gpt-", "o1-", "o3-", "text-", "davinci", "curie", "babbage", "ada"))

def _needs_max_completion_tokens(model_name: str) -> bool:
    """Check if the model requires max_completion_tokens instead of max_tokens.

    Models that need max_completion_tokens:
    - o1, o3 series (reasoning models)
    - gpt-4o, gpt-4o-mini (newer OpenAI models)
    - gpt-5 series (future models)
    """
    model_lower = model_name.lower()
    return (
        model_lower.startswith(("o1-", "o3-", "gpt-5")) or
        "gpt-4o" in model_lower  # Covers gpt-4o, gpt-4o-mini, etc.
    )

def _fix_json_control_characters(json_text: str) -> str:
    """
    Fix unescaped control characters in JSON string values.

    Handles cases where models output literal newlines/tabs in JSON strings instead of escaped versions.
    Example: {"key": "value with\nnewline"} -> {"key": "value with\\nnewline"}
    """
    result = []
    i = 0
    in_string = False
    string_char = None
    escaped = False

    while i < len(json_text):
        char = json_text[i]

        if escaped:
            # Previous char was backslash, keep this char as-is
            result.append(char)
            escaped = False
            i += 1
            continue

        if char == '\\':
            result.append(char)
            escaped = True
            i += 1
            continue

        if not in_string:
            # Not inside a string
            if char in ('"', "'"):
                # Starting a string
                in_string = True
                string_char = char
                result.append(char)
            else:
                result.append(char)
            i += 1
            continue

        # Inside a string
        if char == string_char:
            # Ending the string
            in_string = False
            string_char = None
            result.append(char)
            i += 1
            continue

        # Inside a string - check for control characters
        if char == '\n':
            result.append('\\n')
        elif char == '\r':
            result.append('\\r')
        elif char == '\t':
            result.append('\\t')
        elif char == '\b':
            result.append('\\b')
        elif char == '\f':
            result.append('\\f')
        else:
            result.append(char)

        i += 1

    return ''.join(result)

def extract_json_from_response(response_text: str) -> dict:
    """
    Robustly extract JSON from various response formats.
    Handles:
    - Pure JSON: {...}
    - Markdown code block: ```json\n{...}\n```
    - Text with JSON embedded
    - Unescaped control characters in string values (newlines, tabs, etc.)
    """
    import json
    import re

    if not response_text:
        return {}

    def try_parse(text: str, fix_control_chars: bool = False) -> dict:
        """Try to parse JSON, optionally fixing control characters first."""
        if fix_control_chars:
            text = _fix_json_control_characters(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # If error mentions control characters and we haven't tried fixing yet, return None to trigger retry
            if not fix_control_chars and 'control character' in str(e).lower():
                return None
            return None

    # Strategy 1: Try direct JSON parsing
    result = try_parse(response_text, fix_control_chars=False)
    if result is not None:
        return result

    # Strategy 1b: Try with control character fixing
    result = try_parse(response_text, fix_control_chars=True)
    if result is not None:
        return result

    # Strategy 2: Extract from markdown code block (```json ... ```)
    json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        extracted = json_match.group(1)
        result = try_parse(extracted, fix_control_chars=False)
        if result is not None:
            return result
        result = try_parse(extracted, fix_control_chars=True)
        if result is not None:
            return result

    # Strategy 3: Extract from generic code block (``` ... ```)
    json_match = re.search(r'```\s*\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:
        extracted = json_match.group(1)
        result = try_parse(extracted, fix_control_chars=False)
        if result is not None:
            return result
        result = try_parse(extracted, fix_control_chars=True)
        if result is not None:
            return result

    # Strategy 4: Find JSON object in text (look for {...})
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        extracted = json_match.group(0)
        result = try_parse(extracted, fix_control_chars=False)
        if result is not None:
            return result
        result = try_parse(extracted, fix_control_chars=True)
        if result is not None:
            return result

    # If all strategies fail, return empty dict
    print(f"Warning: Could not extract JSON from response: {response_text[:200]}...")
    return {}

def _get_client(model_name: str) -> OpenAI:
    """Get the appropriate OpenAI client based on model name"""
    if _is_openai_model(model_name):
        # Use OpenAI API for GPT models
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
        return OpenAI(api_key=api_key)
    else:
        # Use OpenRouter for other models (Claude, Gemini, etc.)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for non-OpenAI models")
        openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(
            api_key=api_key,
            base_url=openrouter_base_url
        )

def save_prompts(sys_prompt: str, prompt: str, filename: str = "prompts.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== New Call ===\n")
        f.write("System Prompt:\n")
        f.write(sys_prompt.strip() + "\n\n")
        f.write("User Prompt:\n")
        f.write(prompt.strip() + "\n")
        f.write("="*30 + "\n\n")


class TokenLimitExceededError(Exception):
    """Raised when the context length exceeds the model's token limit"""
    pass


def chat_completion_with_retries(model: str, sys_prompt: str, prompt: str, max_retries: int = 5, retry_interval_sec: int = 20, top_logprobs=0, response_format=None, image_content=None, **kwargs) -> Mapping:
    """
    Chat completion with retries, supporting both text and multimodal (image) content.

    Args:
        model: Model name or local model path (e.g., "/path/to/local/model")
        sys_prompt: System prompt
        prompt: User prompt (text)
        image_content: Optional. Can be either:
                      - A list of content items (for multimodal messages)
                      - A base64 encoded image string
        Other args: Same as before

    Raises:
        TokenLimitExceededError: If the context length exceeds the model's token limit

    Note:
        If model is a local path (starts with "/" or contains "/ndata/", "/home/", etc.),
        it will use local inference with vLLM or transformers instead of API calls.
    """

    # Check if this is a local model
    if _is_local_model(model):
        print(f"[Local Model] Detected local model path: {model}")
        # Note: Local models don't support image content yet
        if image_content:
            print("[Local Model] Warning: Image content is not supported for local models, ignoring images")

        # Extract max_tokens from kwargs
        max_tokens = kwargs.pop("max_tokens", 4096)
        temperature = kwargs.pop("temperature", 0.8)

        for n_attempts_remaining in range(max_retries, 0, -1):
            try:
                return _local_model_chat_completion(
                    model_path=model,
                    sys_prompt=sys_prompt,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_logprobs=top_logprobs,
                    **kwargs
                )
            except Exception as e:
                print(f"[Local Model] Error: {e}")
                print(f"[Local Model] Retrying... ({n_attempts_remaining - 1} attempts remaining)")
                if n_attempts_remaining > 1:
                    time.sleep(retry_interval_sec)
        return {}

    # Get the appropriate client based on model name (API-based models)
    client = _get_client(model)

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            # Build user message content
            if image_content:
                # Multimodal message with image
                if isinstance(image_content, list):
                    # Already formatted as list of content items
                    user_content = [{"type": "text", "text": prompt}] + image_content
                else:
                    # Single image in base64 format
                    user_content = [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_content}",
                                "detail": "high"
                            }
                        }
                    ]
            else:
                # Text-only message
                user_content = prompt

            create_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                **kwargs
            }
            save_prompts(sys_prompt, prompt)
            if top_logprobs > 0:
                create_params["logprobs"] = True
                create_params["top_logprobs"] = top_logprobs

            # Only use response_format for OpenAI models (Claude doesn't support it properly)
            if response_format and _is_openai_model(model):
                create_params["response_format"] = response_format

            # Handle o1, o3, gpt-5 series model restrictions
            if _needs_max_completion_tokens(model):
                # Convert max_tokens to max_completion_tokens
                if "max_tokens" in create_params:
                    create_params["max_completion_tokens"] = create_params.pop("max_tokens")
                # Remove temperature (only default value 1 is supported)
                create_params.pop("temperature", None)

            res = client.chat.completions.create(**create_params)
            # print("LLM's direct response", res.choices[0].message.content)
            return res

        except openai.BadRequestError as e:
            # Check if this is a token limit error
            error_message = str(e)
            if "maximum context length" in error_message.lower() or "reduce the length" in error_message.lower():
                print(f"\n{'='*80}")
                print(f"ERROR: Context length exceeded!")
                print(f"Error: {error_message}")
                print(f"{'='*80}\n")
                # Don't retry for token limit errors - raise immediately
                raise TokenLimitExceededError(f"Context length exceeded: {error_message}") from e
            else:
                # For other BadRequest errors, print and retry
                print(e)
                print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
                time.sleep(retry_interval_sec)
        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
            json.JSONDecodeError,
            ) as e:
            print(e)
            print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    return {}


def truncate_text(text, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"WARNING: Maximum token length exceeded ({len(tokens)} > {max_tokens})")
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def get_embedding_with_retries(text: str, model: str = "text-embedding-ada-002", max_retries: int = 5, retry_interval_sec: int = 20, max_tokens: int = 8000):
    """
    Get text embedding using OpenAI API with retries.
    
    Args:
        text: Text to embed
        model: OpenAI embedding model to use (default: text-embedding-ada-002)
        max_retries: Maximum number of retry attempts
        retry_interval_sec: Seconds to wait between retries
        max_tokens: Maximum number of tokens (text will be truncated if exceeded)
    
    Returns:
        numpy array of embedding vector, or None if failed
    """
    import numpy as np
    
    # Truncate text if too long
    text = truncate_text(text, max_tokens)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        return None
    
    # Use direct OpenAI API (not through openrouter)
    client = OpenAI(api_key=api_key)
    
    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            
            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
        ) as e:
            print(f"OpenAI embedding error: {e}")
            print(f"Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    
    print("Failed to get embedding after all retries")
    return None

