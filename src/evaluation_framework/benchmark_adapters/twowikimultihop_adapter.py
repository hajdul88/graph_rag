import requests
import os
import json
import random
from typing import Optional, Union, Any, LiteralString, List
from .base_benchmark_adapter import BaseBenchmarkAdapter
from tools.embedding import EmbeddingPipeline
import numpy as np


class TwoWikiMultihopAdapter(BaseBenchmarkAdapter):
    dataset_info = {
        "filename": "/app/datasets/2wikimultihop_dev.json",
        "URL": "https://huggingface.co/datasets/voidful/2WikiMultihopQA/resolve/main/dev.json",
    }

    def __init__(self, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initializes the MusiqueQAAdapter with an embedding pipeline and encountered documents.
        """
        self.embedding_pipeline = EmbeddingPipeline(model_name=embedding_model)
        self.encountered_docs = {}

    def _get_corpus_entries(self, item: dict[str, Any]) -> List[str]:
        """Extracts corpus entries from the paragraphs of an item."""
        corpus_list = []
        for paragraph in item.get('context', []):
            title = paragraph[0]
            sentences = " ".join(paragraph[1])
            if title not in self.encountered_docs:
                corpus_list.append(sentences)
                self.encountered_docs[title] = [
                    self.embedding_pipeline.create_embedding(sentences)]
            else:
                similarity = 0
                e1 = self.embedding_pipeline.create_embedding(sentences)
                for e2 in self.encountered_docs[title]:
                    dot_product = np.dot(e1, e2)
                    norm_1 = np.linalg.norm(e1)
                    norm_2 = np.linalg.norm(e2)

                    cur_sim = dot_product / (norm_1 * norm_2)
                    if cur_sim > similarity:
                        similarity = cur_sim
                if similarity < 0.99:
                    corpus_list.append(sentences)
                    self.encountered_docs[title].append(e1)

        return corpus_list

    def load_corpus(
            self, limit: Optional[int] = None, seed: int = 42
    ) -> tuple[list[Union[LiteralString, str]], list[dict[str, Any]]]:
        filename = self.dataset_info["filename"]

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                corpus_json = json.load(f)
        else:
            response = requests.get(self.dataset_info["URL"])
            response.raise_for_status()
            corpus_json = response.json()

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(corpus_json, f, ensure_ascii=False, indent=4)

        if limit is not None and 0 < limit < len(corpus_json):
            random.seed(seed)
            corpus_json = random.sample(corpus_json, limit)

        corpus_list = []
        question_answer_pairs = []
        for dict in corpus_json:
            corpus_list.extend(self._get_corpus_entries(dict))

            question_answer_pairs.append(
                {
                    "question": dict["question"],
                    "answer": dict["answer"].lower(),
                    "type": dict["type"],
                }
            )

        return corpus_list, question_answer_pairs
