# Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems

This repository contains implementation code for a testbed designed to evaluate a novel Retrieval-Augmented Generation (RAG) system proposed in the paper *“Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems.”* The system enhances document retrieval for complex multi-hop reasoning by treating information as an interconnected graph and using spreading-activation signals to traverse relationships and identify relevant evidence.
## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [SA-RAG System Architecture](#sa-rag-system-architecture)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Configuration Guide](#configuration-guide)
  - [Evaluation Configuration](#evaluation-configuration)
  - [Model & LLM Configuration](#model--llm-configuration)
  - [Retrieval & Spreading Activation Parameters](#retrieval--spreading-activation-parameters)
  - [File Paths Configuration](#file-paths-configuration)
  - [Database Configuration](#database-configuration)
- [Run Full Pipeline](#run-full-pipeline)
    - [Monitor Execution](#monitor-execution)
- [Visualization](#visualization)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements a **Graph-based RAG system** that improves document retrieval for complex questions requiring multi-hop reasoning. Unlike traditional RAG systems that treat documents as isolated chunks, this system: 

1. **Creates a Knowledge Graph**: Documents are ingested and converted into a structured knowledge graph stored in Neo4j, with entities, relationships, and document chunks as nodes.

2. **Applies Spreading Activation**: When answering a query, activation signals propagate through the graph based on semantic similarity and graph structure, helping identify relevant documents across multiple hops.

3. **Enables Multi-hop Reasoning**: The system can traverse 2+ hops through the knowledge graph to find evidence and reason over complex, multi-step questions.

4. **Generates Reasoning Traces**: The system produces interpretable reasoning steps showing how it navigated the graph to arrive at answers.

## Key Features
- **NLP Pipeline for Knowledge Graph Creation**: Automates entity and relation extraction for knowledge graph construction.
- **Implementation of Baseline RAG Pipelines**: Includes implementation of several RAG pipelines used in the paper
- **Iterative Retrieval and Reasoning Feature**: Implements multi-step retrieval and reasoning.
- **Query Decomposition Feature**: Supports breaking down multi-hop questions into sub-questions.
- **Spreading Activation Retrieval**: Implements a novel document retrieval method based on the spreading activation algorithm.
- **Flexible Benchmarking**: Supports MuSiQuE and TwoWikiMultiHop datasets.
- **Comprehensive Evaluation**: Provides built-in evaluation metrics (EM, F1) and dashboard generation.
- **Graph Visualization**: Visualizes retrieved subgraphs and spreading activation patterns.
- **Containerized Deployment**: Uses Docker and Docker Compose for reproducible experiments.

## SA-RAG System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY INPUT                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  1. Create Query Embedding     │
        │     (BAAI/bge-large-en-v1.5)   │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  2. Initial Retrieval (k=4 documents)  │
        │     Semantic similarity search in Neo4j│
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────────────┐
        │  3. Spreading Activation (k_hop=3)       │
        │     - Activate initial results           │
        │     - Propagate activation through       │
        │       relationships (3 hops)             │
        │     - Prune inactive nodes               │
        │     - Normalize scores                   │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  4. Multi-hop Reasoning (steps=3)      │
        │     Generate reasoning trace showing   │
        │     how graph was traversed            │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  5. Answer Generation                  │
        │     Combine reasoning + retrieved      │
        │     documents to generate final answer │
        └────────────────┬───────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  ANSWER + REASONING TRACE + RETRIEVED DOCUMENTS                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python**:  3.11 or higher
- **Docker & Docker Compose**: For containerized execution (recommended)
- **Neo4j**: Running via Docker (recommended) or existing instance
- **Ollama**: For running LLM locally or accessible via network
- **Poetry**: Python dependency management (for local installation)

## Installation and Setup

1. **Clone the repository**: 
   ```bash
   git clone https://github.com/hajdul88/graph_rag.git
   cd graph_rag
   ```

⚠️ Before proceeding, ensure the `LLM_ENDPOINT_URL` environment variable is configured in both `docker-compose.yml` and `Dockerfile`.



2. **Start all services** (Neo4j + App):
   ```bash
   docker-compose up -d
   ```

   This will: 
   - Build the Docker image for the application
   - Start two Neo4j instances (primary and secondary)
   - Start the application container

⚠️ To use the app with the MuSiQuE benchmark, download the dataset from [this link](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing), extract the `musique_ans_v1.0_dev.jsonl` file to the `datasets` directory, and rename it to `musique_data.jsonl`.
 
3. **Verify services are running**:
   ```bash
   docker-compose ps
   ```

   Expected output:
   ```
   NAME                COMMAND                  SERVICE             STATUS
   graph_rag-app-1     "poetry run python ..."  app                 Up (healthy)
   graph_rag-neo4j-1   "/sbin/tini -- /startup" neo4j               Up (healthy)
   graph_rag-neo4j_new-1  "/sbin/tini -- /startup" neo4j_new        Up (healthy)
   ```

**Access services**:
- **Neo4j Browser (Primary)**: http://localhost:7474
  - Username: `neo4j`
  - Password: `test1234`
- **Neo4j Browser (Secondary)**: http://localhost:7475
  - Username: `neo4j`
  - Password: `test1234`
- **Application logs**: `docker-compose logs -f app`

5. **Stop services**:
   ```bash
   docker-compose down
   ```

   To also remove data volumes:
   ```bash
   docker-compose down -v
   ```

## Configuration Guide

All configuration is managed through `src/config.py`. Modify the dataclass definitions to customize the pipeline behavior.

### Evaluation Configuration

Control which pipeline stages run and experiment scale:

```python
@dataclass
class EvaluationConfig:
    building_corpus_from_scratch: bool = True      # Build new corpus or skip
    ingest_corpus:  bool = True                      # Ingest corpus into Neo4j
    number_of_samples_in_corpus: int = 1            # Number of benchmark samples to process
    benchmark:  str = "MuSiQuE"                      # 'MuSiQuE', 'TwoWikiMultiHop'
    answering_questions: bool = True                # Generate answers
    evaluating_answers: bool = True                 # Compute evaluation metrics
    evaluation_engine: str = "DeepEval"             # Evaluation framework
    evaluation_metrics: List[str] = ["EM", "f1"]    # Metrics to compute
    dashboard:  bool = True                          # Generate HTML dashboard
    delete_at_end: bool = False                     # Delete Neo4j data after completion
    record_context_graphs: bool = True              # Save retrieved subgraph visualizations
    questions_subset_vis: List[str] = ["2hop__67660_81007"]  # Questions to visualize
``` 

**Key parameters**: 
- `number_of_samples_in_corpus`: ⚠️ Full benchmark can take hours.  Start with 1-5 samples for testing.
- `benchmark`: Choose dataset (MuSiQuE ot TwoWikiMultiHop)
- `delete_at_end`: Set to `False` to preserve Neo4j data for inspection
- `record_context_graphs`: Set to `True` to save graphs for visualization (availabel only for MuSiQuE)

### Model & LLM Configuration

Configure language models, embedding models, and ingestion pipeline:

```python
@dataclass
class LLMConfig:
    # LLM & Embedding Models
    endpoint_url: str = os.environ['OLLAMA_URL']           # LLM API endpoint
    model_name: str = "phi4"                             # LLM model for reasoning/answering
    embedding_model_name: str = "BAAI/bge-large-en-v1.5" # Embedding model for semantic search
    
    # Knowledge Graph Ingestion
    ingestion_type: str = "advanced_knowledge_graph"     # Pipeline type
    chunking:  str = "word_based"                         # Chunking strategy
    chunk_size_ingestion: int = 500                      # Tokens per chunk
    overlap_size_ingestion: int = 100                    # Token overlap between chunks
    NER_template: str = '/app/files/templates/template_ner_v2.txt'    # Entity extraction
    RE_template: str = '/app/files/templates/template_re_v2.txt'      # Relation extraction
    
    # Agent & Reasoning
    agent_type: str = "modified_diffusion_agent"  # See available agents below
    reasoning_enabled: bool = True                # Enable reasoning steps
    reasoning_steps: int = 3                      # Number of reasoning iterations
```

**Available agent types**:
- `modified_diffusion_agent`:  Uses spreading activation (main paper contribution)
- `baseline_cot`: Traditional chain-of-thought RAG without graph traversal
- `decomposition_agent`: Decomposes questions into sub-questions
- `hybrid`: Combines spreading activation and decomposition strategies

### Retrieval & Spreading Activation Parameters

Core parameters controlling graph traversal and activation:

```python
@dataclass
class LLMConfig:
    # ===== GRAPH TRAVERSAL =====
    K_HOP: int = 3                          # K-hop neighborhood around initially activated nodes
                                            # Higher = find more distant evidence
                                            # Lower = faster, better reasoning
    
    
    
    # ===== INITIAL RETRIEVAL =====
    RETRIEVE_K: int = 4                     # Number of initial documents from semantic search
                                            # These become the seed nodes for activation
    
    ACTIVATING_DESCRIPTIONS: int = 4        # Deprecated: Should match RETRIEVE_K
    
    # ===== ACTIVATION THRESHOLDS =====
    ACTIVATION_THRESHOLD:  float = 0.5      # Minimum score for nodes and links to become active
                                            # Lower = keep more nodes, broader context
                                            # Higher = keep only strongest connections
    
    PRUNING_THRESHOLD: float = 0.45         # Filter out documents whose cosine similarity to input query
                                            # is below this threshold after activation

    
    # ===== NORMALIZATION =====
    NORMALIZATION_PARAMETER: float = 0.4    # Normalization constant for edge weights

```

**Example configurations**:

```python
# MuSiQuE Recommended
K_HOP = 4
RETRIEVE_K = 3
ACTIVATING_DESCRIPTIONS = 3
ACTIVATION_THRESHOLD = 0.5
PRUNING_THRESHOLD = 0.45 
NORMALIZATION_PARAMETER = 0.4

# TwoWikiMultiHop Recommended
K_HOP = 3
RETRIEVE_K = 10
ACTIVATING_DESCRIPTIONS = 10
ACTIVATION_THRESHOLD = 0.5
PRUNING_THRESHOLD = 0.45 
NORMALIZATION_PARAMETER = 0.4
```

### File Paths Configuration

Configure input/output locations:

```python
@dataclass
class FilePathConfig:
    questions_base:  str = "/app/files/questions"      # Questions directory
    answers_base: str = "/app/files/answers"          # Answers directory
    results_base: str = "/app/results"                # Results/metrics directory
    templates_folder: str = "/app/files/templates"    # Prompt templates
    graphs_folder: str = "/app/files/graphs"          # Saved graphs
    
    questions_file_name: str = "musique_test"         # Benchmark name
    corpus_file_name: str = "musique_corpus_test"
    answers_file_name: str = "musique_test"
```

### Database Configuration

Neo4j connection settings:

```python
@dataclass
class DatabaseConfig:
    NEO4J_URL: str = os.environ['NEO4J_URL']          # bolt://neo4j:7687 (Docker) or bolt://localhost:7687 (Local)
    NEO4J_USER: str = os.environ['NEO4J_USER']        # neo4j
    NEO4J_PASSWORD:  str = os.environ['NEO4J_PASSWORD'] # test1234
```

**Tips**:
- Use `NEO4J_URL` (primary instance) for experiments requiring clean data
- Use `NEO4J_NEW_URL` (secondary instance) to preserve existing data

## Run Full Pipeline

```bash
docker-compose up
# Pipeline runs automatically
# View logs: docker-compose logs -f app
```

The pipeline automatically executes in order: 

1. **Corpus Building** (if `building_corpus_from_scratch=True`)
   - Loads benchmark datasets
   - Creates corpus JSON file

2. **Corpus Ingestion** (if `ingest_corpus=True`)
   - Chunks documents (configurable:  500 tokens, 100 overlap)
   - Extracts entities and relations
   - Creates text embeddings
   - Creates knowledge graph in Neo4j

3. **Question Answering** (if `answering_questions=True`)
   - Loads questions from file and runs answering logic based on selected RAG pipeline

4. **Evaluation** (if `evaluating_answers=True`)
   - Computes metrics (EM, F1)
   - Saves results to JSON

5. **Dashboard Generation** (if `dashboard=True`)
   - Creates HTML dashboard with metrics
   - Saves to `results/{answers_file_name}_dashboard.html`

6. **Graph Recording** (if `record_context_graphs=True`)
   - Saves retrieved subgraphs as pickle files
   - Useful for understanding retrieval process

7. **Cleanup** (if `delete_at_end=True`)
   - Deletes Neo4j data

### Monitor Execution

```bash
# Docker
docker-compose logs -f app
```

**Example Output:**

```
2026-01-06 11:05:00 - INFO - Corpus Builder started... 
2026-01-06 11:05:15 - INFO - Building corpus from MuSiQuE benchmark...
2026-01-06 11:05:30 - INFO - Corpus Builder Ended... 
2026-01-06 11:05:30 - INFO - Ingestion execution time: 00:00:30
2026-01-06 11:05:30 - INFO - Creating database index...
2026-01-06 11:05:35 - INFO - INGESTION FINISHED
2026-01-06 11:05:35 - INFO - Question answering started...
2026-01-06 11:06:00 - INFO - Loaded 1 questions from /app/files/questions/musique_test_questions.json
2026-01-06 11:06:15 - INFO - Question answering ended...
2026-01-06 11:06:15 - INFO - Question answering execution time: 00:00:40
2026-01-06 11:06:15 - INFO - Evaluation started...
2026-01-06 11:06:30 - INFO - Evaluation ended...
2026-01-06 11:06:30 - INFO - Evaluation execution time: 00:00:15
2026-01-06 11:06:30 - INFO - Dashboard generated successfully
```

## Visualization

### Graph Visualization Notebook

Use `graph_visualization.ipynb` to visualize retrieved subgraphs:

```python
import pickle
import networkx as nx
from graph_visualization import visualize_graph, visualize_diffusion

# Load saved graph
with open("files/graphs/2hop__67660_81007_retrieved.pkl", "rb") as f:
    G = pickle.load(f)

# Visualize retrieval
visualize_graph(G, title='Retrieved Subgraph')

# Visualize spreading activation
with open("files/graphs/2hop__67660_81007_diffusion_0.3_0.5. pkl", "rb") as f:
    G = pickle.load(f)
visualize_diffusion(G, title='Spreading Activation Pattern')
```

**Color legend**:
- 🔴 **Red**: Activated nodes (by spreading activation)
- 🟡 **Yellow**: Golden answer nodes (from dataset)
- 🩷 **Pink**: Both activated and golden
- 🔵 **Light Blue**: Other nodes
- 🟠 **Orange edges**: Relevant relationships
- ⚫ **Gray edges**: Other relationships

## Citation

If you use this code, please cite the paper:

```bibtex
@article{pavlovic2025leveraging,
  title={Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems},
  author={Pavlovi{\'c}, Jovan and Kr{\'e}sz, Mikl{\'o}s and Hajdu, L{\'a}szl{\'o}},
  journal={arXiv preprint arXiv:2512.15922},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details