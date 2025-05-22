# SpecSearcherAI

A RAG-powered AI assistant for searching and retrieving information from educational documents.

## Overview

SpecSearcherAI is a Retrieval-Augmented Generation (RAG) tool that allows users to search through educational materials such as course specifications and textbooks. Built with Chainlit and AutoGen, this application provides a conversational interface where users can ask questions about course content and receive accurate, context-aware responses.

## Features

- **Document Retrieval**: Search through indexed course materials with natural language queries
- **Conversational Interface**: Chat-based UI powered by Chainlit
- **Persistent Memory**: Maintains conversation history between sessions
- **Vector Search**: Uses ChromaDB for efficient semantic search
- **Support for Multiple Document Types**: Indexes PDFs, DOCX, and text files
- **Azure OpenAI Integration**: Leverages Azure's AI services for natural language understanding

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database (for user sessions)
- Azure OpenAI API access

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/LumpyStructure/SpecSearcherAI.git
   cd SpecSearcherAI
   ```

2. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```

3. Create a .env file based on .env.example and fill in your credentials
   ```bash
   cp .env.example .env
   ```

4. Index your documents
   ```bash
   python reindex_docs.py
   ```

## Usage

1. Start the application
   ```bash
   python -m chainlit run app_v2.py
   ```

2. Open your browser and navigate to `http://localhost:8000`

3. Ask questions about the indexed educational materials

## Architecture

- **app_v2.py**: Main application file with Chainlit UI integration
- **spec_surfer_custom_no_class.py**: Core functionality for Azure OpenAI integration and memory management
- **file_indexer.py**: Document processing and chunking for vector database indexing
- **reindex_docs.py**: Script to (re)index all documents in the source directory

## Contributing

This project is currently not accepting contributions.

## License

This project is licensed under the MIT License.