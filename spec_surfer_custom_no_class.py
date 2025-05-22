import os
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
)
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from file_indexer import SimpleDocumentIndexer
from dotenv import load_dotenv


# Function to create an Azure OpenAI client
def get_azure_openai_client(model=None, max_tokens=3000, temperature=0, seed=42):
    return AzureOpenAIChatCompletionClient(
        azure_deployment=model if model else os.getenv("MODEL_DEPLOYMENT_NAME"),
        model=model if model else os.getenv("MODEL"),
        api_version=(
            os.getenv("MODEL_API_VERSION")
            if os.getenv("MODEL_API_VERSION")
            else "2025-01-01-preview"
        ),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )


async def index_docs() -> None:
    indexer = SimpleDocumentIndexer(memory=rag_memory, chunk_size=1000)
    sources = [
        r"C:\Users\frase\OneDrive\Documents\RINA work experience\AI Agents\specifications\AQA-7516-7517-SP-2015.PDF",
        r"C:\Users\frase\OneDrive\Documents\RINA work experience\AI Agents\specifications\PG A Level  Computer Science.pdf",
    ]
    chunks: int = await indexer.index_documents(sources)
    print(f"Indexed {chunks} chunks from {len(sources)} documents")


def initialise_vector_memory():
    # Initialize vector memory
    rag_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="comp_sci_docs",
            persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
            k=10,  # Return top 10 results
            score_threshold=0.4,  # Minimum similarity score
        )
    )
    # # Uncomment the following lines to re-index documents
    # await rag_memory.clear()  # Clear existing memory
    # print("Indexing documents...")
    # rag_memory = await index_docs()

    return rag_memory


load_dotenv()
# Initialize the vector memory
rag_memory = initialise_vector_memory()

# Create our RAG assistant agent
rag_assistant = AssistantAgent(
    name="rag_assistant",
    model_client=get_azure_openai_client(),
    memory=[rag_memory],
    system_message="""You are a helpful AI Assistant.
    The files you have been provided with contains information for a computer science course.
    When given a user query, use these files to help the user with their request.
    If you cannot find the answer in the files, respond that it is not part of the course.""",
    model_client_stream=True,
)