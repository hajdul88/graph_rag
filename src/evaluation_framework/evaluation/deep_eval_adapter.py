from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evaluation_framework.evaluation.base_eval_adapter import BaseEvalAdapter
from evaluation_framework.evaluation.metrics.exact_match import ExactMatchMetric
from evaluation_framework.evaluation.metrics.f1 import F1ScoreMetric
from typing import Any, Dict, List


class DeepEvalAdapter(BaseEvalAdapter):
    def __init__(self):
        self.g_eval_metrics = {
            "EM": ExactMatchMetric(),
            "f1": F1ScoreMetric()
        }

    async def evaluate_answers(
        self, answers: List[Dict[str, Any]], evaluator_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        for metric in evaluator_metrics:
            if metric not in self.g_eval_metrics:
                raise ValueError(f"Unsupported metric: {metric}")

        results = []
        for answer in answers:
            test_case = LLMTestCase(
                input=answer["question"],
                actual_output=answer["answer"],
                expected_output=answer["golden_answer"],
            )
            metric_results = {}
            for metric in evaluator_metrics:
                metric_to_calculate = self.g_eval_metrics[metric]
                metric_to_calculate.measure(test_case)
                metric_results[metric] = {
                    "score": metric_to_calculate.score,
                    "reason": metric_to_calculate.reason,
                }
            results.append({**answer, "metrics": metric_results})

        return results
