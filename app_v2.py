from typing import cast, Dict, Optional
import chainlit as cl
import os
import sqlite3
from chainlit.types import ThreadDict
from chainlit.input_widget import Select
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from spec_surfer_custom_no_class import (
    get_azure_openai_client,
    initialise_vector_memory,
)
from dotenv import load_dotenv


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    return default_user


async def get_collections():
    connection = sqlite3.connect("C:/Users/frase/.chromadb_autogen/chroma.sqlite3")
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM collections")
    names = cursor.fetchall()
    connection.close()
    return [name[0] for name in names]


async def select_folder():
    names = await get_collections()
    rag_select = await cl.AskActionMessage(
        content="Select documents",
        actions=[
            cl.Action(name=name, payload={"value": name}, label=name) for name in names
        ],
    ).send()

    return rag_select.get("payload").get("value")


@cl.on_chat_start
async def startup():
    folder_name = await select_folder()

    assistant = AssistantAgent(
        name="rag_assistant",
        model_client=get_azure_openai_client(),
        memory=[await initialise_vector_memory(folder_name)],
        system_message=os.getenv("SYSTEM_MESSAGE"),
        model_client_stream=True,
    )
    # Set the assistant agent in the user session.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("agent", assistant)  # type: ignore


@cl.on_message
async def main(message: cl.Message):
    # Get the assistant agent from the user session.
    spec_surfer_agent = cast(AssistantAgent, cl.user_session.get("agent"))  # type: ignore
    response = cl.Message(content="")
    async for message in spec_surfer_agent.on_messages_stream(
        messages=[TextMessage(content=message.content, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(message, ModelClientStreamingChunkEvent):
            # Stream the model client response to the user.
            await response.stream_token(message.content)
        elif isinstance(message, Response):
            # Done streaming the model client response. Send the message.
            await response.send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    folder_name = await select_folder()

    rag_memory = await initialise_vector_memory(folder_name)
    user_memory = ListMemory()

    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            await user_memory.add(
                MemoryContent(content=message["output"], mime_type=MemoryMimeType.TEXT)
            )
        else:
            await user_memory.add(
                MemoryContent(content=message["output"], mime_type=MemoryMimeType.TEXT)
            )

    assistant = AssistantAgent(
        name="rag_assistant",
        model_client=get_azure_openai_client(),
        memory=[rag_memory, user_memory],
        system_message=os.getenv("SYSTEM_MESSAGE"),
        model_client_stream=True,
    )

    cl.user_session.set("memory", assistant._memory)
    cl.user_session.set("agent", assistant)  # type: ignore


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    load_dotenv()
    run_chainlit(__file__)
