# Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems

This repository contains the experimental code for the paper **"Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems"**. The project implements a novel Retrieval-Augmented Generation (RAG) system that enhances document retrieval for complex multi-hop reasoning questions by treating information as interconnected graphs and using activation signals to traverse relationships and find related evidence.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start with Docker](#quick-start-with-docker)
- [Configuration Guide](#configuration-guide)
  - [Evaluation Configuration](#evaluation-configuration)
  - [Model & LLM Configuration](#model--llm-configuration)
  - [Retrieval & Spreading Activation Parameters](#retrieval--spreading-activation-parameters)
  - [File Paths Configuration](#file-paths-configuration)
  - [Database Configuration](#database-configuration)
- [Running Experiments](#running-experiments)
- [Pipeline Stages](#pipeline-stages)
- [Advanced Configuration](#advanced-configuration)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project implements a **Graph-based Retrieval-Augmented Generation (RAG)** system that improves document retrieval for complex questions requiring multi-hop reasoning. Unlike traditional RAG systems that treat documents as isolated chunks, this system: 

1. **Creates a Knowledge Graph**: Documents are ingested and converted into a structured knowledge graph stored in Neo4j, with entities, relationships, and document chunks as nodes.

2. **Applies Spreading Activation**: When answering a query, activation signals propagate through the graph based on semantic similarity and graph structure, helping identify relevant documents across multiple hops.

3. **Enables Multi-hop Reasoning**: The system can traverse 2+ hops through the knowledge graph to find evidence and reason over complex, multi-step questions.

4. **Generates Reasoning Traces**: The system produces interpretable reasoning steps showing how it navigated the graph to arrive at answers.

### Key Concepts

- **Spreading Activation**: An algorithm that simulates activation spreading through a semantic network, where nodes become active based on initial query relevance and propagate activation to connected nodes. 
- **Knowledge Graph**: A structured representation of documents where entities are nodes and relationships (e.g., "mentions", "answers") are edges.
- **Multi-hop Reasoning**: The ability to connect evidence across multiple documents to answer complex questions.
- **Neo4j Database**: A graph database optimized for storing and querying knowledge graphs efficiently.

## Key Features

- ✅ **Spreading Activation Retrieval**: Novel activation-based document retrieval using graph traversal
- ✅ **Multi-hop Reasoning**: Answer questions requiring evidence from multiple documents
- ✅ **Flexible Benchmarking**: Support for MuSiQuE, HotPotQA, and TwoWikiMultiHop datasets
- ✅ **Advanced NLP Pipeline**: Automatic entity and relation extraction for knowledge graph construction
- ✅ **Configurable Agents**: Multiple RAG agent implementations (diffusion-based, baseline, hybrid, decomposition)
- ✅ **Comprehensive Evaluation**: Built-in evaluation metrics (EM, F1) and dashboard generation
- ✅ **Graph Visualization**:  Visualize retrieved subgraphs and spreading activation patterns
- ✅ **Containerized Deployment**: Docker and Docker Compose for reproducible experiments

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY INPUT                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  1. Create Query Embedding     │
        │     (BAAI/bge-large-en-v1.5)  │
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
        │  3. Spreading Activation (k_hop=3)      │
        │     - Activate initial results          │
        │     - Propagate activation through      │
        │       relationships (3 hops)            │
        │     - Prune inactive nodes              │
        │     - Normalize scores                  │
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
        │  5. Answer Generation                   │
        │     Combine reasoning + retrieved       │
        │     documents to generate final answer  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  ANSWER + REASONING TRACE + RETRIEVED DOCUMENTS               │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python**:  3.11 or higher
- **Docker & Docker Compose**: For containerized execution
- **Neo4j**: Running via Docker (recommended) or existing instance
- **Ollama**: For running LLM locally or accessible via network

## Installation

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/hajdul88/graph_rag.git
cd graph_rag

# Start all services (Neo4j + App)
docker-compose up
```

The `docker-compose.yml` starts:
- **neo4j**: Primary Neo4j instance (port 7687 for Bolt, 7474 for Web UI)
- **neo4j_new**: Secondary Neo4j instance (port 7688 for Bolt, 7475 for Web UI)
- **app**: Main application container

### Option 2: Local Installation

```bash
git clone https://github.com/hajdul88/graph_rag. git
cd graph_rag

# Create virtual environment
python3. 11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using Poetry
pip install poetry==1.6.1
poetry install

# Ensure Neo4j is running on bolt://localhost:7687
# Ensure Ollama is running (or accessible at configured URL)
```

## Quick Start with Docker

1. **Start services**:
   ```bash
   docker-compose up -d
   ```

2. **Run experiments**:
   The container automatically runs `src/run_eval.py` which executes the entire pipeline: 
   ```bash
   # View logs
   docker-compose logs -f app
   ```

3. **Access Neo4j Browser**:
   - Primary instance: http://localhost:7474 (user: neo4j, password: test1234)
   - Secondary instance:  http://localhost:7475

4. **View results**:
   ```bash
   # Generated files in ./results/
   ls results/
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
    benchmark:  str = "MuSiQuE"                      # 'MuSiQuE', 'HotPotQA', 'TwoWikiMultiHop'
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
- `number_of_samples_in_corpus`: Start with 1-5 for testing, increase for full experiments
- `benchmark`: Choose dataset (MuSiQuE recommended, multi-hop questions)
- `delete_at_end`: Set to `False` to preserve Neo4j data for inspection

### Model & LLM Configuration

Configure language models, embedding models, and ingestion pipeline:

```python
@dataclass
class LLMConfig:
    # LLM & Embedding Models
    ollama_url: str = os.environ['OLLAMA_URL']           # Ollama service endpoint
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
- `modified_diffusion_agent`: **Recommended** - Uses spreading activation (main paper contribution)
- `baseline_cot`: Traditional chain-of-thought RAG without graph traversal
- `hybrid`: Combines diffusion and decomposition strategies
- `decomposition_agent`: Decomposes questions into sub-questions

### Retrieval & Spreading Activation Parameters

Core parameters controlling graph traversal and activation:

```python
@dataclass
class LLMConfig:
    # ===== GRAPH TRAVERSAL =====
    K_HOP: int = 3                          # Number of hops for spreading activation
                                            # Higher = broader search, more compute
                                            # Lower = focused search, faster
    
    # ===== INITIAL RETRIEVAL =====
    RETRIEVE_K: int = 4                     # Number of initial documents from semantic search
                                            # These become the seed nodes for activation
    
    ACTIVATING_DESCRIPTIONS: int = 4        # Number of candidate nodes activated per hop
                                            # Controls branching factor in graph traversal
    
    # ===== ACTIVATION THRESHOLDS =====
    ACTIVATION_THRESHOLD:  float = 0.5       # Minimum score for a node to remain active
                                            # Lower = keep more nodes, broader context
                                            # Higher = keep only strongest connections
    
    PRUNING_THRESHOLD: float = 0.45         # Score below which nodes are removed
                                            # Should be < ACTIVATION_THRESHOLD
                                            # Removes weakly activated nodes from memory
    
    # ===== NORMALIZATION =====
    NORMALIZATION_PARAMETER: float = 0.4    # Decay factor for activation across hops
                                            # Lower = faster decay (emphasis on close nodes)
                                            # Higher = slower decay (broader influence)
```

**Tuning Guide**: 

| Parameter | Increase | Decrease |
|-----------|----------|----------|
| `K_HOP` | Find distant evidence | Speed up / reduce hallucination |
| `RETRIEVE_K` | Broader initial context | Focus on most relevant only |
| `ACTIVATING_DESCRIPTIONS` | Explore more connections | Reduce branching factor |
| `ACTIVATION_THRESHOLD` | Only keep strong links | Keep more weak connections |
| `NORMALIZATION_PARAMETER` | Distant nodes stay relevant | Emphasize nearby nodes |

**Example configurations**:

```python
# ✅ Balanced (Default)
K_HOP = 3
RETRIEVE_K = 4
ACTIVATING_DESCRIPTIONS = 4
ACTIVATION_THRESHOLD = 0.5
NORMALIZATION_PARAMETER = 0.4

# 🚀 Fast (fewer hops, focused)
K_HOP = 2
RETRIEVE_K = 3
ACTIVATING_DESCRIPTIONS = 2
ACTIVATION_THRESHOLD = 0.6
NORMALIZATION_PARAMETER = 0.3

# 🔍 Comprehensive (more hops, broader search)
K_HOP = 4
RETRIEVE_K = 6
ACTIVATING_DESCRIPTIONS = 6
ACTIVATION_THRESHOLD = 0.4
NORMALIZATION_PARAMETER = 0.5
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
    NEO4J_URL: str = os.environ['NEO4J_URL']          # bolt://neo4j:7687
    NEO4J_USER: str = os.environ['NEO4J_USER']        # neo4j
    NEO4J_PASSWORD: str = os. environ['NEO4J_PASSWORD'] # test1234
```

**Tips**:
- Use `NEO4J_URL` (primary instance) for experiments requiring clean data
- Use `NEO4J_NEW_URL` (secondary instance) to preserve existing data
- Change in Docker environment variables if using local Neo4j

## Running Experiments

### Run Full Pipeline

```bash
cd graph_rag

# Using Docker Compose (recommended)
docker-compose up

# Using local installation
cd src
python run_eval.py
```

The pipeline automatically executes in order: 

1. **Corpus Building** (if `building_corpus_from_scratch=True`)
   - Downloads benchmark dataset (MuSiQuE, HotPotQA, etc.)
   - Creates corpus JSON file

2. **Corpus Ingestion** (if `ingest_corpus=True`)
   - Chunks documents (configurable: 500 tokens, 100 overlap)
   - Extracts entities and relations (via NER/RE)
   - Creates knowledge graph in Neo4j

3. **Question Answering** (if `answering_questions=True`)
   - Loads questions from file
   - For each question: 
     - Creates embedding
     - Retrieves initial documents
     - Applies spreading activation
     - Generates reasoning steps
     - Produces final answer

4. **Evaluation** (if `evaluating_answers=True`)
   - Computes metrics (EM, F1)
   - Saves results to JSON

5. **Dashboard Generation** (if `dashboard=True`)
   - Creates HTML dashboard with metrics
   - Saves to `results/{answers_file_name}_dashboard.html`

6. **Graph Recording** (if `record_context_graphs=True`)
   - Saves retrieved subgraph visualizations
   - Useful for understanding retrieval process

7. **Cleanup** (if `delete_at_end=True`)
   - Deletes Neo4j data

### Monitor Execution

```bash
# View logs (Docker)
docker-compose logs -f app

# View logs (local)
# Logs are printed to console

# Access Neo4j Browser
# Primary:  http://localhost:7474
# Secondary: http://localhost:7475
```

### Example Output

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

## Pipeline Stages

### Stage 1: Corpus Building

**File**: `src/evaluation_framework/corpus_builder/`

Builds a question-answer corpus from a benchmark dataset: 
- Downloads from HuggingFace (MuSiQuE, HotPotQA, TwoWikiMultiHop)
- Extracts questions and golden answers
- Saves to `/app/files/questions/`

**Control**:  `EvaluationConfig. building_corpus_from_scratch`

### Stage 2: Corpus Ingestion

**File**: `src/ingestion/`

Converts corpus into knowledge graph: 

1. **Chunking**: Splits documents into 500-token chunks with 100-token overlap
2. **Embedding**: Creates semantic embeddings for each chunk
3. **NER**: Extracts named entities using LLM + template
4. **RE**: Extracts relationships between entities using LLM + template
5. **Storage**: Stores in Neo4j as nodes and relationships

**Control**: `EvaluationConfig.ingest_corpus`, `LLMConfig.chunk_size_ingestion`

### Stage 3: Question Answering

**File**: `src/retrieval/`

Answers questions using spreading activation:

1. **Embedding**: Creates query embedding
2. **Initial Retrieval**: Semantic search for top-4 relevant documents
3. **Spreading Activation**:
   - Initializes activation on retrieved documents
   - Propagates activation through graph (3 hops)
   - Prunes low-score nodes based on thresholds
   - Normalizes scores using decay parameter
4. **Reasoning**: Generates interpretable multi-step reasoning
5. **Answer Generation**: LLM produces final answer

**Control**: `LLMConfig.agent_type`, spreading activation parameters

### Stage 4: Evaluation

**File**: `src/evaluation_framework/evaluation/`

Computes metrics comparing generated answers to golden answers: 
- **EM** (Exact Match): Binary match on normalized strings
- **F1**:  Token-level overlap between answer and golden answer

**Control**: `EvaluationConfig.evaluation_metrics`

### Stage 5: Dashboard Generation

**File**: `src/tools/summarization. py`

Creates interactive HTML dashboard with:
- Aggregate metrics (average EM, F1)
- Per-question breakdown
- Visualization of retrieval process

**Output**: `results/{answers_file_name}_dashboard.html`

**Control**: `EvaluationConfig.dashboard`

### Stage 6: Graph Recording (Optional)

**File**: `src/tools/context_recorder.py`

Saves retrieved subgraph visualizations for analysis:
- Records which documents were activated
- Shows activation levels and connections
- Useful for understanding retrieval decisions

**Control**: `EvaluationConfig.record_context_graphs`

## Advanced Configuration

### Experiment 1: Compare Agent Types

Test different retrieval strategies:

```python
# In src/config.py, modify LLMConfig. agent_type: 

# Run 1: Spreading Activation (Paper Contribution)
agent_type = "modified_diffusion_agent"

# Run 2: Baseline without graph
agent_type = "baseline_cot"

# Run 3: Hybrid approach
agent_type = "hybrid"

# Run 4: Question decomposition
agent_type = "decomposition_agent"
```

### Experiment 2: Tune Activation Parameters

Find optimal spreading activation settings:

```python
# Conservative (fewer false positives)
ACTIVATION_THRESHOLD = 0.6
PRUNING_THRESHOLD = 0.55
K_HOP = 2
RETRIEVE_K = 3

# Aggressive (broader search)
ACTIVATION_THRESHOLD = 0.4
PRUNING_THRESHOLD = 0.3
K_HOP = 4
RETRIEVE_K = 6
```

### Experiment 3: Change Embedding Models

Test different embedding quality:

```python
# High-quality but slower
embedding_model_name = "BAAI/bge-large-en-v1.5"

# Fast but less precise
embedding_model_name = "all-MiniLM-L6-v2"

# Domain-specific
embedding_model_name = "msmarco-distilbert-base-v3"
```

### Experiment 4: Scale to Full Benchmark

Run on all questions: 

```python
# In src/config.py
EvaluationConfig.number_of_samples_in_corpus = 100  # or -1 for all
```

⚠️ **Warning**: Full benchmark can take hours.  Start with 1-5 samples for testing.

### Experiment 5: Different Benchmarks

```python
# MuSiQuE (recommended, multi-hop)
benchmark = "MuSiQuE"

# HotPotQA (multi-hop)
benchmark = "HotPotQA"

# TwoWikiMultiHop (2-hop questions)
benchmark = "TwoWikiMultiHop"
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

## Troubleshooting

### Neo4j Connection Issues

```
Error: Could not connect to bolt://neo4j:7687
```

**Solutions**:
1. Verify Neo4j is running:  `docker-compose ps`
2. Check Neo4j logs: `docker-compose logs neo4j`
3. Wait for startup (Neo4j can take 10-15 seconds): `sleep 20 && docker-compose up app`
4. Verify credentials in `docker-compose.yml` match `src/config.py`

### Ollama Connection Issues

```
Error: Connection refused to OLLAMA_URL
```

**Solutions**:
1. Check Ollama is running on configured URL
2. Update `OLLAMA_URL` in `docker-compose.yml`
3. Pull required model: `ollama pull phi4`

### Out of Memory

**Solutions**:
1. Reduce `number_of_samples_in_corpus`
2. Reduce `K_HOP` (fewer graph hops)
3. Reduce `RETRIEVE_K` (fewer initial documents)
4. Increase Docker memory limit:  `--memory 8g` in docker-compose.yml

### Slow Ingestion

**Solutions**:
1. Check Neo4j index creation: `SHOW INDEXES` in Neo4j Browser
2. Reduce `chunk_size_ingestion` to process faster
3. Increase Docker resources

### Empty Results

**Causes**:
- Benchmark dataset not found
- Neo4j empty (corpus not ingested)
- Semantic search not returning results

**Solutions**:
1. Check files:  `ls files/questions/`
2. Verify ingestion: query Neo4j Browser
3. Increase `RETRIEVE_K` for less restrictive search

## Performance Tips

1. **First Run**: Expect 5-15 minutes for corpus ingestion + first question
2. **Subsequent Runs**:  Use `building_corpus_from_scratch = False` to skip ingestion
3. **Preserve Data**: Set `delete_at_end = False` to keep Neo4j between runs
4. **Parallel Processing**: Currently single-threaded; multi-threading possible in `answer_generation_executor. py`

## Citation

If you use this code, please cite the paper:

```bibtex
@article{spreading_activation_rag,
  title={Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems},
  author={Hajdul, ... },
  year={2026}
}
```

## License

MIT License - See LICENSE file for details