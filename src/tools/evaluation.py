from datetime import datetime
from .loggers import create_file_loger
import pandas as pd
import csv
from typing import List, Any
import time
import re
import string


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def generate_answers(agent: Any, queries_df: pd.DataFrame) -> pd.DataFrame:
    """Generates answers for a set of queries using the provided agent.

        Args:
            agent: The RAG agent instance used for generating answers.
            queries_df (pd.DataFrame): DataFrame containing queries and their types.
                Expected columns: ['query', 'question_type']

        Returns:
            pd.DataFrame: DataFrame containing original queries, question types, and generated answers.
                Columns: ['query', 'question_type', 'answer']
        """
    queries = []
    query_types = []
    llm_answers = []
    logger = create_file_loger(f'answers_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    for index, (query, q_type) in queries_df[['query', 'type']].iterrows():
        answer = agent.generate_answer(query)
        llm_answers.append(answer)
        queries.append(query)
        query_types.append(q_type)
        logger.info(f"{index}\nQuery: {query}\nType: {q_type}\nAnswer: {answer}\n")
        time.sleep(0.5)
    return pd.DataFrame({'query': queries, 'question_type': query_types, 'answer': llm_answers})


def retrieve_documents(agent: Any, queries_df: pd.DataFrame, at_k: int = 5) -> List[List[str]]:
    """Retrieves relevant documents for a set of queries using the provided agent.

        Args:
            agent: The RAG agent instance used for document retrieval.
            queries_df (pd.DataFrame): DataFrame containing queries.
                Expected columns: ['query']
            at_k (int, optional): Number of documents to retrieve per query. Defaults to 5.

        Returns:
            List[List[str]]: A list where each inner list contains the retrieved documents
                for a single query. Each inner list has length at_k.

        Note:
            Creates a log file with timestamp containing query-document pairs for debugging.
            Log format: 'retrieval_YYYY-MM-DD_HH-MM-SS.log'
        """
    retrieved = []
    logger = create_file_loger(f'retrieval_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    for query in queries_df['query'].tolist():
        docs = agent.retrieve_relevant_docs(query, at_k)
        logger.info(f'Query: {query}\nDocs: {docs}\n')
        retrieved.append(docs)

    return retrieved


def evaluate_qa(golden: List[str], llm: List[str], result_file: str) -> None:
    """Evaluates question-answering performance by comparing golden answers with LLM-generated answers.

      Args:
          golden (List[str]): List of correct (golden) answers.
          llm (List[str]): List of LLM-generated answers to evaluate.
          result_file (str): Path to the file where evaluation results will be written.

      Raises:
          ValueError: If the lengths of golden and llm answer lists don't match.

      Notes:
          Precision is calculated as the number of correct answers divided by total questions.
          An answer is considered correct if the golden answer is a substring of the LLM answer.
      """
    logger = create_file_loger(f'evaluation_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    if len(golden) != len(llm):
        raise ValueError()
    total = len(golden)
    correct = 0
    for a, b in zip(golden, llm):
        logger.info(f'\nGolden: {a}\nLLM: {b}\n---------------------------')
        if _normalize_answer(a) in _normalize_answer(b):
            correct += 1
    precision = correct / total
    with open(result_file, 'w') as f:
        f.write(f'Precision: {precision}')


def evaluate_retrieval(golden: List[List[str]], llm: List[List[str]], k: int,
                       result_file: str) -> None:
    """Evaluates retrieval performance by comparing golden and LLM retrieved documents.

    Args:
        golden (List[List[str]]): List of lists containing ground truth document snippets for each query
        llm (List[List[str]]): List of lists containing LLM retrieved document snippets for each query
        k (int): Number of top documents to consider in the evaluation
        result_file (str): Path to the file where evaluation metrics will be written

    Returns:
        None: Writes evaluation metrics to the specified file in CSV format with the following metrics:
            - hits@k: Proportion of queries with at least one relevant document in top k results
            - map@k: Mean Average Precision at k across all queries
            - mrr@k: Mean Reciprocal Rank at k across all queries

    Notes:
        - Documents are normalized by removing spaces and newlines before comparison
        - MAP calculation considers precision at each relevant document position
        - MRR uses the reciprocal of the first relevant document's rank
    """

    assert len(golden) == len(llm)
    n_queries = len(golden)
    metrics = {
        f'hits_at_{k}': 0,
        f'map_at_{k}': 0,
        f'mrr_at_{k}': 0
    }
    hits_at_k_sum = 0
    average_precision_sum = 0
    reciprocal_rank_sum = 0
    for golden_docs, llm_docs in zip(golden, llm):
        golden_docs = [item.replace(" ", "").replace("\n", "") for item in golden_docs]
        llm_docs = [item.replace(" ", "").replace("\n", "") for item in llm_docs]
        hit_flag = False
        first_relevant_rank = 1000
        relevant_items = 0
        precision_at_rank_sum = 0
        for rank, ret_doc in enumerate(llm_docs[:k]):
            if any(gold_doc == ret_doc for gold_doc in golden_docs):
                hit_flag = True
                first_relevant_rank = min(rank + 1, first_relevant_rank)
                relevant_items += 1
                precision_at_rank_sum += relevant_items / (rank + 1)

        hits_at_k_sum += int(hit_flag)
        average_precision_sum += precision_at_rank_sum / relevant_items \
            if not relevant_items == 0 else 0
        reciprocal_rank_sum += 1 / first_relevant_rank \
            if first_relevant_rank < 1000 else 0

    metrics[f'hits_at_{k}'] = hits_at_k_sum / n_queries
    metrics[f'map_at_{k}'] = average_precision_sum / n_queries
    metrics[f'mrr_at_{k}'] = reciprocal_rank_sum / n_queries

    with open(result_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerows([metrics])
