import re
from typing import List
from PyPDF2 import PdfReader
from docx import Document
import aiofiles
import aiohttp
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType


class SimpleDocumentIndexer:
    """Basic document indexer for AutoGen Memory."""

    def __init__(self, memory: Memory, chunk_size: int = 1500) -> None:
        self.memory = memory
        self.chunk_size = chunk_size
        # self.persist_directory = r"C:\Users\frase\OneDrive\Documents\RINA work experience\AI Agents\chroma_dbs"

    async def _fetch_content(self, source: str) -> str:
        """Fetch content from URL or file."""
        if source.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    return await response.text()
        elif source.casefold().endswith(".pdf"):
            # Handle PDF files
            reader = PdfReader(source)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif source.casefold().endswith(".docx"):
            # Handle DOCX files
            doc = Document(source)
            text = ""
            for para in doc.paragraphs:
                text += para.text
            return text
        else:
            async with aiofiles.open(source, "r") as f:
                return await f.read()

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks: list[str] = []
        # Just split text into fixed-size chunks
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk.strip())
        return chunks

    async def index_documents(self, sources: List[str]) -> int:
        """Index documents into memory."""
        total_chunks = 0

        for source in sources:

            # embedding = OpenAIEmbedding
            # if os.path.exists(rf"C:\Users\frase\OneDrive\Documents\RINA work experience\AI Agents\chroma_dbs\{source.split(os.sep)[-1]}"):
            #     print(f"File {source} already indexed.")
            #     self.memory.add(Chroma(
            #         persist_directory=persist_directory, embedding_function=embedding
            #     ))
            try:
                content = await self._fetch_content(source)

                # Strip HTML if content appears to be HTML
                if "<" in content and ">" in content:
                    content = self._strip_html(content)

                chunks = self._split_text(content)

                for i, chunk in enumerate(chunks):
                    await self.memory.add(
                        MemoryContent(
                            content=chunk,
                            mime_type=MemoryMimeType.TEXT,
                            metadata={"source": source, "chunk_index": i},
                        )
                    )

                total_chunks += len(chunks)

            except Exception as e:
                print(f"Error indexing {source}: {str(e)}")

        return total_chunks
