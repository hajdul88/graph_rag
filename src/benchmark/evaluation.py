from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import pandas as pd


def evaluate_retriever(corpus_path, queries_path, qrels_path, retriever, metrics_k_values):
    """
        Evaluates retriever performance using standard IR metrics.

        Metrics calculated:
        - NDCG@k
        - MAP@k
        - Recall@k
        - Precision@k
        - F1@k

        Args:
            corpus_path: Path to document corpus
            queries_path: Path to evaluation queries
            qrels_path: Path to relevance judgments
            retriever: Retriever instance to evaluate
            metrics_k_values: List of k values for metrics

        Returns:
            pandas.DataFrame: Evaluation metrics and their values
    """

    # Step 1: Load evaluation dataset components
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path,
        query_file=queries_path,
        qrels_file=qrels_path).load_custom()

    results = {}
    cnt = 0

    # Step 2: Execute retrieval for each query
    for key, query in queries.items():
        # Retrieve top document for query
        docs_with_score = retriever.retrieve(query, top_k=1)
        # Store retrieval results for each query
        results[key] = {
            doc["file_name"]: doc['similarity']
            for doc in docs_with_score
        }
        cnt += 1

    # Step 3: Calculate evaluation metrics
    ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, metrics_k_values
    )
    # Step 4: Compile metrics into dictionary
    metrics_dict = {}
    for k in metrics_k_values:
        metrics_dict[f"NDCG@{k}"] = ndcg[f"NDCG@{k}"]
        metrics_dict[f"MAP@{k}"] = map_[f"MAP@{k}"]
        metrics_dict[f"Recall@{k}"] = recall[f"Recall@{k}"]
        metrics_dict[f"Precision@{k}"] = precision[f"P@{k}"]
        metrics_dict[f"F1@{k}"] = 2 * precision[f"P@{k}"] * recall[f"Recall@{k}"] / (precision[f"P@{k}"] + recall[f"Recall@{k}"])

    return pd.DataFrame({'metrics': metrics_dict.keys(), 'values': metrics_dict.values()})
