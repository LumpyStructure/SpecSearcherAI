import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

def get_gemini_client(model=None, max_tokens=3000, temperature=0):
    return OpenAIChatCompletionClient(
        model=model if model else os.getenv("MODEL"),
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("GEMINI_BASE_URL"),
        max_tokens=max_tokens,
        temperature=temperature,
        model_info={
            "family": model if model else os.getenv("MODEL"),
            "function_calling": False,
            "json_output": False,
            "vision": False,
            "multiple_system_messages": True,
        },
    )
