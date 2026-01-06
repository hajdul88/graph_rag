import json
import logging
from typing import Dict, List

import numpy as np
from neo4j import GraphDatabase
from ollama import Client

from .base import RAGAgent
from .graph_retrieval_mixin import GraphRetrievalMixin
from .retrieval_prompt_manager import PromptManager
from .retrieval_config import AgentConfig
from tools.embedding import EmbeddingPipeline
from tools.llm_output import ModelResponse

logger = logging.getLogger(__name__)

reasoning_schema = {
    "type": "object",
    "properties": {
        "original_question": {"type": "string"},
        "subquestions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 1},
                    "question": {"type": "string", "minLength": 1}
                },
                "required": ["id", "question"],
                "additionalProperties": False
            }
        }
    },
    "required": ["original_question", "subquestions"],
    "additionalProperties": False
}


class HybridAgentCoT(RAGAgent, GraphRetrievalMixin):
    """Hybrid agent combining question decomposition with graph-based retrieval. 

    This agent decomposes complex questions into subquestions, retrieves context
    for each using graph diffusion, and combines answers for the final response.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the HybridAgentCoT.

        Args:
            config: AgentConfig instance with all configuration.
        """
        super().__init__(config)
        self.client = Client(host=config.ollama_url)
        self.driver = GraphDatabase.driver(
            config.neo4j.url,
            auth=(config.neo4j.username, config.neo4j.password)
        )
        self.embedding_pipeline = EmbeddingPipeline(config.embedding.model_name)

        prompt_manager = PromptManager()
        self.answering_prompt, self.reasoning_prompt = prompt_manager.get_prompts(
            config.answering_prompt_loc,
            config.reasoning_prompt_loc
        )

        # Graph parameters
        self.k_hop = config.retrieval.k_hop
        self.normalization_parameter = config.retrieval.normalization_parameter

    def _decompose_question(self, query: str) -> List[str]:
        """Decompose a complex question into subquestions. 

        Args:
            query: The original question. 

        Returns:
            List of subquestion strings.
        """
        messages = [
            {"role": "system", "content": self.reasoning_prompt},
            {"role": "user", "content": query}
        ]

        reasoning_result = self.client.chat(
            model=self.config.model_name,
            messages=messages,
            stream=False,
            keep_alive=0,
            format=reasoning_schema,
            options={"temperature": 0.0}
        )

        reasoning_response = json.loads(reasoning_result['message']['content'])
        sub_queries = [q['question'] for q in reasoning_response['subquestions']]

        logger.debug(f"Decomposed question into {len(sub_queries)} subquestions")
        return sub_queries

    def _answer_subquestion(self, subquestion: str, memory: List[str],
                            retrieve_k: int, activating_descriptions: int,
                            activation_threshold: float, pruning_threshold: float) -> str:
        """Generate answer for a single subquestion with accumulated context.

        Args:
            subquestion: The subquestion to answer.
            memory: List of previous question-answer pairs.
            retrieve_k: Number of seed entities to retrieve.
            activating_descriptions: Number of descriptions for activation.
            activation_threshold: Entity score threshold. 
            pruning_threshold:  Relation similarity threshold.

        Returns:
            Formatted string with the subquestion and its answer.
        """
        # Extend question with previous answers for context
        if memory:
            extended_question = f"{subquestion}\n\n" + '\n\n'.join(memory)
        else:
            extended_question = subquestion

        query_embedding = self.embedding_pipeline.create_embedding(extended_question)
        context = self._knowledge_acquisition_step(
            query_embedding,
            retrieve_k,
            activating_descriptions,
            activation_threshold,
            pruning_threshold
        )

        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": subquestion}
        ]

        result = self.client.chat(
            model=self.config.model_name,
            messages=messages,
            stream=False,
            keep_alive=0,
            format=ModelResponse.model_json_schema()
        )
        response = ModelResponse.validate(json.loads(result['message']['content']))

        return f"Sub-question:  {subquestion}\nAnswer: {response.final_answer}"

    def _generate_answer(self, original_question: str, sub_questions: List[str],
                         retrieve_k: int, activating_descriptions: int,
                         activation_threshold: float, pruning_threshold: float) -> Dict:
        """Generate final answer from subquestions and original question.

        Args:
            original_question: The original question. 
            sub_questions: List of decomposed subquestions.
            retrieve_k: Number of seed entities to retrieve.
            activating_descriptions: Number of descriptions for activation. 
            activation_threshold: Entity score threshold.
            pruning_threshold: Relation similarity threshold.

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys. 
        """
        memory = []

        # Answer each subquestion iteratively
        for i, subquestion in enumerate(sub_questions):
            answer_str = self._answer_subquestion(
                subquestion,
                memory[: i],  # Pass only previous answers, not current one
                retrieve_k,
                activating_descriptions,
                activation_threshold,
                pruning_threshold
            )
            memory.append(answer_str)
            logger.debug(f"Answered subquestion {i + 1}/{len(sub_questions)}")

        # Retrieve context for original question
        query_embedding = self.embedding_pipeline.create_embedding(original_question)
        context = self._knowledge_acquisition_step(
            query_embedding,
            retrieve_k,
            activating_descriptions,
            activation_threshold,
            pruning_threshold
        )

        # Combine with subquestion answers
        final_context = (
                f"{context}\n\n**Answers to sub-questions**\n\n"
                + '\n\n'.join(memory)
        )

        # Generate final answer
        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": final_context},
            {"role": "user", "content": original_question}
        ]

        result = self.client.chat(
            model=self.config.model_name,
            messages=messages,
            stream=False,
            keep_alive=0,
            format=ModelResponse.model_json_schema()
        )
        response = ModelResponse.validate(json.loads(result['message']['content']))

        return {
            'answer': response.final_answer,
            'reasoning': response.reasoning,
            'knowledge': final_context
        }

    def generate_answer(self, query: str, retrieve_k: int = None,
                        activating_descriptions: int = None,
                        activation_threshold: float = None,
                        pruning_threshold: float = None) -> Dict:
        """Generate answer using hybrid decomposition and graph retrieval.

        Args:
            query: The user's question.
            retrieve_k: Number of seed entities (uses config default if not specified).
            activating_descriptions:  Descriptions for activation (uses config default if not specified).
            activation_threshold: Entity threshold (uses config default if not specified).
            pruning_threshold: Relation threshold (uses config default if not specified).

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys. 
        """
        retrieve_k = retrieve_k or self.config.retrieval.retrieve_k
        activating_descriptions = activating_descriptions or self.config.retrieval.activating_descriptions
        activation_threshold = activation_threshold or self.config.retrieval.activation_threshold
        pruning_threshold = pruning_threshold or self.config.retrieval.pruning_threshold

        # Decompose the question
        sub_queries = self._decompose_question(query)

        # Generate and return answer
        return self._generate_answer(
            query,
            sub_queries,
            retrieve_k,
            activating_descriptions,
            activation_threshold,
            pruning_threshold
        )