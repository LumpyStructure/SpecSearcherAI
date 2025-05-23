import asyncio
import os
from dotenv import load_dotenv
from spec_surfer_custom_no_class import initialise_vector_memory, index_docs


async def main():
    load_dotenv()
    rag_memory = await initialise_vector_memory(
        os.path.normpath(os.getenv("RAG_SOURCES")).split(os.sep)[-1]
    )
    rag_memory = await index_docs(rag_memory)
    return rag_memory


if __name__ == "__main__":
    asyncio.run(main())
