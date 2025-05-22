import asyncio
from spec_surfer_custom_no_class import initialise_vector_memory, index_docs

async def main():
    rag_memory = await initialise_vector_memory()
    rag_memory = await index_docs(rag_memory)
    return rag_memory

if __name__ == "__main__":
    asyncio.run(main())
    