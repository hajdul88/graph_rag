# Graph RAG Ingestion Pipeline

This project implements the **ingestion** part of a **Graph-based Retrieval-Augmented Generation (RAG)** pipeline. The goal is to asynchronously read documents, chunk the content, and store the resulting relationships in a **Neo4j** graph database for efficient retrieval.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [CI/CD](#ci-cd)
- [Contributing](#contributing)
- [License](#license)

## Overview

In a Graph RAG pipeline, the ingestion phase is responsible for reading and processing large documents, breaking them into manageable chunks, and creating a graph representation. This project handles the ingestion phase, where it reads files asynchronously, chunks the data, and creates nodes and relationships in a **Neo4j** graph database.

### Key Concepts:

- **Asynchronous File Reading**: Efficiently read large files in parallel to improve ingestion performance.
- **Chunking**: Break large documents into smaller chunks to facilitate retrieval during the generation phase.
- **Graph Creation**: Store chunks as nodes and create relationships based on content similarity or structure within the document.
- **Neo4j Database**: The chunks and relationships are stored in a Neo4j graph database, enabling efficient querying and retrieval for RAG tasks.

## Features

- Asynchronous file reading for efficient document ingestion.
- Chunking mechanism to split large documents into smaller, retrievable sections.
- Neo4j graph database integration to store document chunks as nodes and relationships.
- Support for multiple document types (e.g., text, PDF).
- Configurable chunking logic to suit different content structures.
- Basic CI/CD setup with Docker and GitHub Actions.

## Architecture

The ingestion pipeline is designed to handle large document ingestion and is built with the following components:

- **Asynchronous Ingestion**: Files are read asynchronously using Python's `asyncio` to improve throughput.
- **Chunking Logic**: Document contents are chunked based on configurable parameters (e.g., max characters per chunk, semantic boundaries).
- **Neo4j Storage**: Chunked data is stored in a Neo4j graph database, where chunks are represented as nodes and related content as relationships.

### Workflow:

1. **File Ingestion**: Documents are ingested asynchronously.
2. **Chunking**: Each document is split into smaller chunks.
3. **Graph Creation**: Nodes (chunks) and edges (relationships) are created in Neo4j.
4. **Querying**: The stored chunks can be queried during the RAG phase to augment LLM-based generation tasks.

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker (optional but recommended)
- Neo4j (can be run locally via Docker or connected to a cloud-hosted instance)

### Install Dependencies

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/hajdul88/graph-rag.git
cd graph-rag
