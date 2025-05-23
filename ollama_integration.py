from autogen_ext.models.ollama import OllamaChatCompletionClient
import os


def get_ollama_client(model=None, max_tokens=3000, temperature=0, seed=42):
    return OllamaChatCompletionClient(
        model=model if model else os.getenv("MODEL"),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
