"""Modified Diffusion BFS Agent implementation."""
import json
import time
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

reasoning_schema = {
    "type": "object",
    "properties": {
        "provided_context": {"type": "string"},
        "answer_possible": {"type": "boolean"},
        "final_answer": {"type": "string"},
        "additional_question": {"type": "string"},
    },
    "required": ["provided_context", "answer_possible", "final_answer", "additional_question"]
}


class DiffusionBFSAgent(RAGAgent, GraphRetrievalMixin):
    """Agent that performs knowledge graph traversal using diffusion-based BFS.

    This agent retrieves relevant entities and relationships from a knowledge graph,
    performs a diffusion process to identify relevant information, and generates an answer.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the DiffusionBFSAgent. 

        Args:
            config: AgentConfig instance with all configuration. 
        """
        super().__init__(config)
        self.client = Client(host=config.llm_endpoint_url)
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

        # Graph retrieval parameters
        self.k_hop = config.retrieval.k_hop
        self.normalization_parameter = config.retrieval.normalization_parameter

    def _generate_answer(self, query: str, context: str) -> Dict[str, str]:
        """Generate an answer using the LLM, optionally with reasoning loops. 

        Args:
            query: Original user query.
            context: Retrieved and diffused context string.

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys. 
        """
        query_prompt = "**Question:  **" + query

        if self.config.reasoning.enabled:
            context = self._apply_reasoning_loop(query_prompt, context)

        return self._llm_final_answer(query_prompt, context)

    def _apply_reasoning_loop(self, query: str, context: str) -> str:
        """Apply iterative reasoning to refine context. 

        Args:
            query: The formatted query string.
            context: Initial context from retrieval.

        Returns:
            Refined context after reasoning steps.
        """
        for _ in range(self.config.reasoning.max_steps):
            messages = [
                {"role": "system", "content": self.reasoning_prompt},
                {"role": "user", "content": query},
                {"role": "user", "content": context}
            ]

            try:
                reasoning_result = self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    stream=False,
                    options={"temperature": self.config.reasoning.temperature},
                    keep_alive=0,
                    format=reasoning_schema,
                )
                reasoning_response = json.loads(reasoning_result['message']['content'])
            except Exception as e:
                print(f"Reasoning step failed: {e}")
                continue

            if reasoning_response['answer_possible']:
                break

            # Refine context with additional question
            summarized_context = reasoning_response['provided_context']
            additional_question = reasoning_response['additional_question']
            query_embedding = self.embedding_pipeline.create_embedding(additional_question)
            new_context = self._knowledge_acquisition_step(
                query_embedding,
                self.config.retrieval.retrieve_k,
                self.config.retrieval.activating_descriptions,
                self.config.retrieval.activation_threshold,
                self.config.retrieval.pruning_threshold
            )
            context = (f"# Known information\n{summarized_context}\n"
                       f"# Additional facts{new_context[12:]}")

        return context

    def _llm_final_answer(self, query: str, context: str) -> Dict[str, str]:
        """Query LLM for final answer with retry logic. 

        Args:
            query: The formatted query string.
            context: The context to use for generation.

        Returns:
            Dictionary with answer, reasoning, and knowledge. 
        """
        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": query}
        ]

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                result = self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    stream=False,
                    keep_alive=0,
                    format=ModelResponse.model_json_schema(),
                    options={"temperature": self.config.reasoning.temperature}
                )
                response = ModelResponse.validate(json.loads(result['message']['content']))
                return {
                    'answer': response.final_answer,
                    'reasoning': response.reasoning,
                    'knowledge': context
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)

    def generate_answer(self, query: str, retrieve_k: int = None, **kwargs) -> Dict[str, str]:
        """Generate an answer for the given query. 

        Args:
            query: The user's natural language question.
            retrieve_k: Number of seed entities (uses config default if not specified).
            **kwargs: Additional parameters for advanced usage. 

        Returns:
            A dict containing 'answer', 'reasoning', and 'knowledge'. 
        """
        retrieve_k = retrieve_k or self.config.retrieval.retrieve_k

        query_embedding = self.embedding_pipeline.create_embedding(query)
        context = self._knowledge_acquisition_step(
            query_embedding,
            retrieve_k,
            self.config.retrieval.activating_descriptions,
            self.config.retrieval.activation_threshold,
            self.config.retrieval.pruning_threshold
        )
        return self._generate_answer(query, context)