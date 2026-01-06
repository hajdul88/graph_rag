from typing import Any
from retrieval.base import RAGAgent


class AnswerGeneratorExecutor:
    def __init__(self, rag_agent: RAGAgent):
        self.rag_pipeline = rag_agent

    def question_answering_non_parallel(
            self, questions: list[dict[str, Any]]
    ):
        if not questions:
            raise ValueError("Questions list cannot be empty")

        answers = []

        for instance in questions:
            query_text = instance["question"]
            correct_answer = instance["answer"]

            search_results = self.rag_pipeline.generate_answer(query_text)
            answers.append(
                {
                    "question": query_text,
                    "knowledge": search_results['knowledge'],
                    "reasoning": search_results['reasoning'],
                    "answer": search_results['answer'],
                    "golden_answer": correct_answer
                }
            )

        return answers
