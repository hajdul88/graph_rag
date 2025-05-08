import requests
import os
import json
import random
from typing import Optional, Union, Any, LiteralString
from .base_benchmark_adapter import BaseBenchmarkAdapter


class HotpotQAAdapter(BaseBenchmarkAdapter):
    dataset_info = {
        "filename": "/app/datasets/hotpot_benchmark.json",
        "url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    }

    def load_corpus(
        self, limit: Optional[int] = None, seed: int = 42
    ) -> tuple[list[Union[LiteralString, str]], list[dict[str, Any]]]:
        filename = self.dataset_info["filename"]

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                corpus_json = json.load(f)
        else:
            response = requests.get(self.dataset_info["url"])
            response.raise_for_status()
            corpus_json = response.json()

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(corpus_json, f, ensure_ascii=False, indent=4)

        if limit is not None and 0 < limit < len(corpus_json):
            random.seed(seed)
            corpus_json = random.sample(corpus_json, limit)

        corpus_list = []
        question_answer_pairs = []
        for item in corpus_json:
            for title, sentences in item["context"]:
                corpus_list.append(" ".join(sentences))

            question_answer_pairs.append(
                {
                    "question": item["question"],
                    "answer": item["answer"].lower(),
                    "level": item["level"],
                }
            )

        return corpus_list, question_answer_pairs
