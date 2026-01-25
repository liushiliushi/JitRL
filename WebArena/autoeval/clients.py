import os
import base64
import openai
import numpy as np
from PIL import Image
from typing import Union, Optional
from openai import OpenAI, ChatCompletion

# Configure API key
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")

def _is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI model (should use OpenAI API directly)"""
    return model_name.startswith(("gpt-", "o1-", "o3-", "text-", "davinci", "curie", "babbage", "ada"))


def _needs_max_completion_tokens(model_name: str) -> bool:
    """Check if the model requires max_completion_tokens instead of max_tokens."""
    model_lower = model_name.lower()
    return (
        model_lower.startswith(("o1-", "o3-", "gpt-5")) or
        "gpt-4o" in model_lower
    )


def _supports_temperature(model_name: str) -> bool:
    """Check if the model supports the temperature parameter.

    Some models like GPT-5-mini and o1/o3 reasoning models don't support temperature=0.
    """
    model_lower = model_name.lower()
    # o1, o3, and gpt-5 models don't support temperature parameter
    if model_lower.startswith(("o1-", "o3-", "gpt-5")):
        return False
    return True

def _get_client(model_name: str) -> OpenAI:
    """Get the appropriate OpenAI client based on model name"""
    if _is_openai_model(model_name):
        # Use OpenAI API for GPT models
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
        return OpenAI(api_key=api_key)
    else:
        # Use OpenRouter for other models (Claude, Gemini, etc.)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for non-OpenAI models")
        openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(
            api_key=api_key,
            base_url=openrouter_base_url
        )

# Global client for backward compatibility (defaults to OpenAI)
client = OpenAI()


class LM_Client:
    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        self.model_name = model_name
        # Get the appropriate client based on model name
        self.client = _get_client(model_name)

    def chat(self, messages, json_mode: bool = False) -> tuple[str, ChatCompletion]:
        """
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hi"},
        ])
        """
        # Build API parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
        }
        if json_mode:
            api_params["response_format"] = {"type": "json_object"}
        # Only set temperature for models that support it
        if _supports_temperature(self.model_name):
            api_params["temperature"] = 0

        chat_completion = self.client.chat.completions.create(**api_params)
        response = chat_completion.choices[0].message.content
        return response, chat_completion

    def one_step_chat(
        self, text, system_msg: str = None, json_mode=False
    ) -> tuple[str, ChatCompletion]:
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode)


class GPT4V_Client:
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens
        # Get the appropriate client based on model name
        self.client = _get_client(model_name)

    def encode_image(self, path: str):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def one_step_chat(
        self, text, image: Union[Image.Image, np.ndarray],
        system_msg: Optional[str] = None,
    ) -> tuple[str, ChatCompletion]:
        jpg_base64_str = self.encode_image(image)
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages += [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{jpg_base64_str}"},},
                ],
        }]
        # Build API parameters
        api_params = {
            "model": self.model_name,
            "messages": messages,
        }
        # Use max_completion_tokens for newer models (gpt-4o, o1, etc.)
        if _needs_max_completion_tokens(self.model_name):
            api_params["max_completion_tokens"] = self.max_tokens
        else:
            api_params["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**api_params)
        return response.choices[0].message.content, response


CLIENT_DICT = {
    "gpt-3.5-turbo": LM_Client,
    "gpt-4": LM_Client,
    # "gpt-4o": GPT4V_Client,
    "gpt-4o": LM_Client,
}