import os
import json
import random
from typing import Optional, Any, List, Union, Tuple
from .base_benchmark_adapter import BaseBenchmarkAdapter


class MusiqueQAAdapter(BaseBenchmarkAdapter):
    """Adapter for the Musique QA dataset with local file loading and optional download."""

    dataset_info = {
        "filename": "/app/datasets/musique_ans_v1.0_dev.jsonl",
        "download_url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing"
    }

    def _get_golden_context(self, item: dict[str, Any]) -> str:
        """Extracts golden context from question decomposition and supporting paragraphs."""
        golden_context = []
        paragraphs = item.get("paragraphs", [])

        # Process each decomposition step
        for step in item.get("question_decomposition", []):
            # Add the supporting paragraph if available
            support_idx = step.get("paragraph_support_idx")
            if isinstance(support_idx, int) and 0 <= support_idx < len(paragraphs):
                para = paragraphs[support_idx]
                golden_context.append(f"{para['title']}: {para['paragraph_text']}")

            # Add the step's question and answer
            golden_context.append(f"Q: {step['question']}")
            golden_context.append(f"A: {step['answer']}")
            golden_context.append("")  # Empty line between steps

        return "\n".join(golden_context)

    def _get_raw_corpus(self) -> List[dict[str, Any]]:
        """Loads the raw corpus data from file or downloads it if needed."""
        target_filename = self.dataset_info["filename"]

        if not os.path.exists(target_filename):
            raise NotImplementedError("Download option currently not implemented for this dataset")

        with open(target_filename, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        return data

    def _get_corpus_entries(self, item: dict[str, Any]) -> List[str]:
        """Extracts corpus entries from the paragraphs of an item."""
        return [paragraph["paragraph_text"] for paragraph in item.get("paragraphs", [])]

    def _get_question_answer_pair(
            self,
            item: dict[str, Any],
            load_golden_context: bool = False,
    ) -> dict[str, Any]:
        """Extracts a question-answer pair from an item."""
        qa_pair = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", "").lower()
            if isinstance(item.get("answer"), str)
            else item.get("answer"),
        }

        if load_golden_context:
            qa_pair["golden_context"] = self._get_golden_context(item)

        return qa_pair

    def load_corpus(
            self,
            limit: Optional[int] = None,
            seed: int = 42,
            load_golden_context: bool = True,
    ) -> Tuple[List[str], List[dict[str, Any]]]:
        """Loads and processes the Musique QA dataset with optional filtering."""
        raw_corpus = self._get_raw_corpus()

        if limit is not None and 0 < limit < len(raw_corpus):
            random.seed(seed)
            raw_corpus = random.sample(raw_corpus, limit)

        corpus_list = []
        question_answer_pairs = []

        for item in raw_corpus:
            corpus_list.extend(self._get_corpus_entries(item))
            question_answer_pairs.append(self._get_question_answer_pair(item, load_golden_context))

        return corpus_list, question_answer_pairs
