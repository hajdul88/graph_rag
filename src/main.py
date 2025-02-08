import pandas as pd
from chunk_processor import neo4j_ingestion
from dataset_reader.reader import DatasetReader
import os
from tools.evaluation import (generate_answers, evaluate_qa, evaluate_retrieval, retrieve_documents)
from tools.loggers import console_logger
from agents.naive import BasicMultiHopAgent, BaselineAgent
from tools.neo4j_tools import create_index
import ast


def run_ingestion_pipeline(corpus_document: str = None,
                           mode: str = 'word',
                           chunk_size: int = 500,
                           overlap_size: int = 100,
                           basic_graph_rag: bool = False,
                           db_url: str = None,
                           db_user: str = None,
                           db_pw: str = None,
                           graph_id: int = 0):
    corps_location = f"/app/datasets/{corpus_document}.csv"
    # Process benchmark knowledge documents
    corpus = pd.read_csv(corps_location)

    dataset_reader = DatasetReader(dataframe=corpus, mode=mode, chunk_size=chunk_size, overlap_size=overlap_size,
                                   summarize=basic_graph_rag)

    if basic_graph_rag:
        chunk_processor = neo4j_ingestion.DocumentGraphProcessor(db_url, db_user, db_pw, graph_id=graph_id)
    else:
        chunk_processor = neo4j_ingestion.BasicChunkProcessor(db_url, db_user, db_pw, graph_id=graph_id)
    # ingest chunks in Neo4j
    chunk_processor.process_chunks(dataset_reader)
    # create db index
    create_index(db_url, db_user, db_pw)


def evaluate_agent(agent, agent_name, dataset_name, suffix):
    data_dir = f"/app/datasets/{dataset_name}"
    qa = pd.read_csv(os.path.join(data_dir, f'qa_{suffix}.csv'))
    #agent_response = generate_answers(agent, qa)
    #evaluate_qa(qa['answer'].tolist(), agent_response['answer'].tolist(),
    #            f'/app/results/qa_results_{dataset_name}_{agent_name}.txt')
    qa['evidence_list'] = qa['evidence_list'].apply(ast.literal_eval)
    golden_docs = qa['evidence_list'].tolist()
    llm_list = retrieve_documents(agent, qa, 5)
    evaluate_retrieval(golden_docs, llm_list, 5,
                       f'/app/results/retrieval_{dataset_name}_{agent_name}_{suffix}.txt')


def main():
    NEO4J_URL = os.environ['NEO4J_URL']
    NEO4J_USER = os.environ['NEO4J_USER']
    NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']
    GRAPH_IDS = [3, 2]

    CHUNKING_MODE = 'word'
    CHUNK_SIZE = 500
    OVERLAP_SIZE = 100
    DATASET_NAMES = ['musique', 'hotpotqa']
    SUFFIX = "hippo"

    INGESTION = False
    EVALUATION = True

    console_logger.info('Started')

    if INGESTION:
        for dataset_name, graph_id in zip(DATASET_NAMES, GRAPH_IDS):
            corpus_document = os.path.join(dataset_name, 'corpus' + '_' + SUFFIX)  # corpus dataset document
            console_logger.info(f'Ingesting documents {dataset_name}')
            run_ingestion_pipeline(corpus_document, CHUNKING_MODE, CHUNK_SIZE, OVERLAP_SIZE, False, NEO4J_URL,
                                   NEO4J_USER, NEO4J_PASSWORD, graph_id)
    if EVALUATION:
        for dataset_name, graph_id in zip(DATASET_NAMES, GRAPH_IDS):
            console_logger.info(f'Evaluating retrieval {dataset_name}')
            agent = BaselineAgent(graph_id=graph_id, neo4j_url=NEO4J_URL, neo4j_username=NEO4J_USER,
                                  neo4j_pw=NEO4J_PASSWORD)
            evaluate_agent(agent, 'baseline', dataset_name, SUFFIX)


if __name__ == "__main__":
    main()
