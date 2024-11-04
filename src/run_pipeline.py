import os
import time
import asyncio
from directory_reader import DirectoryFileReader
from chunk_to_network import neo4j_processor as neo4j_ingestion
from benchmark.datasets_beir import download_and_process_data
from retrieval.retriever import DummyRetriever
from benchmark.evaluation import evaluate_retriever
from datetime import datetime


# This is only here for the local setup, otherwise its not needed.
def init():
    """
        Initialize environment variables for local development setup.
        Sets default values for:
        - Directory paths
        - Chunking parameters
        - Neo4j connection details
    """
    os.environ['txt_directory'] = "../example_data/text"
    os.environ['mode'] = "size"
    os.environ['chunk_size'] = "300"
    os.environ['overlap_size'] = "20"
    os.environ['txt_separator'] = "\n\n"
    os.environ['NEO4J_URI'] = "bolt://localhost:7687"
    os.environ['NEO4J_USER'] = "neo4j"
    os.environ['NEO4J_PASSWORD'] = "test1234"


async def main():
    """
        Main pipeline execution function that orchestrates:
        1. Dataset preparation from BEIR benchmark
        2. Document chunking and processing
        3. Neo4j database ingestion
        4. Retrieval evaluation
        5. Saving the results

        Required Environment Variables:
            txt_directory (str): Base directory path for text files
            mode (str): Chunking strategy - either 'size' or 'separator'
            chunk_size (int): Number of characters/tokens per chunk
            overlap_size (int): Number of overlapping characters between chunks
            txt_separator (str): Custom separator for text chunking (e.g. "\n\n")
            NEO4J_URI (str): URI for Neo4j database connection
            NEO4J_USER (str): Username for Neo4j authentication
            NEO4J_PASSWORD (str): Password for Neo4j authentication
            EMBED_MODEL (str): Name of the transformer model for embeddings

        Constants:
            DATASET_NAME (str): Name of the BEIR dataset to use (default: "nfcorpus")
            DATASETS (str): Directory to store downloaded BEIR datasets
            FILES (str): Directory to store processed files
            BENCHMARK_DIRECTORY (str): Full path to benchmark dataset files
        """

    # Load configuration from environment variable
    directory = os.environ['txt_directory']
    mode = os.environ['mode']
    chunk_size = int(os.environ['chunk_size'])
    overlap_size = int(os.environ['overlap_size'])
    txt_separator = os.environ['txt_separator']
    NEO4J_URI = os.environ['NEO4J_URI']
    NEO4J_USER = os.environ['NEO4J_USER']
    NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']
    EMBED_MODEL = os.environ['EMBED_MODEL']
    # Set up benchmark dataset paths
    DATASET_NAME = "nfcorpus"
    DATASETS = "/app/datasets"
    FILES = "/app/files"
    BENCHMARK_DIRECTORY = os.path.join(FILES, DATASET_NAME)

    # Step 1: Download and prepare benchmark dataset
    print("Downloading dataset...")
    download_and_process_data(DATASET_NAME, DATASETS, FILES)

    # Step 2: Initialize document reader with chunking parameters
    print("Reading files and processing chunks...")
    directory_reader = DirectoryFileReader(BENCHMARK_DIRECTORY, mode=mode, chunk_size=chunk_size,
                                           txt_separator=txt_separator, overlap_size=overlap_size)

    # Step 3: Set up Neo4j processor and ingest chunks
    print("Ingesting data into database...")
    chunk_processor = neo4j_ingestion.ChunkProcessor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBED_MODEL)
    await chunk_processor.process_chunks(directory_reader)
    chunk_processor.close()

    # Step 4: Initialize retriever for evaluation
    retriever = DummyRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBED_MODEL)
    # Step 5: Set up evaluation paths
    CORPUS_PATH = os.path.join(DATASETS, DATASET_NAME, 'corpus.jsonl')
    QUERIES_PATH = os.path.join(DATASETS, DATASET_NAME, 'queries.jsonl')
    QRELS_PATH = os.path.join(DATASETS, DATASET_NAME, 'qrels', 'test.tsv')
    # Step 6: Run evaluation
    print("Evaluating retriever")
    results = evaluate_retriever(CORPUS_PATH, QUERIES_PATH, QRELS_PATH, retriever, [1])
    print(F"EVALUATION FINISHED; DATASET {DATASET_NAME}; RESULTS:")
    print(results)
    # Step 7: Save evaluation results
    if not os.path.isdir(os.path.join('/app/results', DATASET_NAME)):
        os.mkdir(os.path.join('/app/results', DATASET_NAME))

    results.to_csv(os.mkdir(
        os.path.join('/app/results', DATASET_NAME, f'evaluation-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')),
        index=False)


if __name__ == "__main__":
    # init() # To run locally
    asyncio.run(main())
    time.sleep(20)
