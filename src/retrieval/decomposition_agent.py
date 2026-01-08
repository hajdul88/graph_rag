"""Decomposition Agent for question decomposition and sequential retrieval."""
import json
import logging
from typing import Dict, List

import torch
from neo4j import GraphDatabase
from ollama import Client

from .base import RAGAgent
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


class DecompositionAgent(RAGAgent):
    """Agent that decomposes complex questions and answers iteratively.

    This agent breaks down a question into simpler subquestions, retrieves
    context for each sequentially, and synthesizes a final answer by combining
    the subquestion answers with context for the original question.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the DecompositionAgent. 

        Args:
            config:  AgentConfig instance with all configuration. 
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

    def _retrieve_documents(self, query_embedding: torch.Tensor, k: int = 5) -> List[str]:
        """Retrieve the most relevant document chunks from Neo4j.

        Args:
            query_embedding:  Tensor containing the embedding vector of the query.
            k: Number of top documents to retrieve.

        Returns:
            List of text strings representing the retrieved document chunks.
        """
        query = """
            WITH $query_embedding AS query_embedding 
            CALL db.index.vector.queryNodes('textEmbedding', 500, query_embedding) 
            YIELD node AS c, score AS similarity
            RETURN c.text AS text, similarity
            ORDER BY similarity DESC
            LIMIT $top_k
        """
        with self.driver.session() as session:
            query_result = session.run(query, query_embedding=query_embedding, top_k=k)
            return [d['text'] for d in query_result.data()]

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
                            retrieve_k: int) -> str:
        """Answer a single subquestion using retrieved context and previous answers.

        Args:
            subquestion: The subquestion to answer.
            memory: List of previous question-answer pairs. 
            retrieve_k: Number of documents to retrieve.

        Returns:
            Formatted string with the subquestion and its answer.
        """
        query_embedding = self.embedding_pipeline.create_embedding(subquestion)
        documents = self._retrieve_documents(query_embedding, retrieve_k)
        context = '\n\n'.join(documents)

        # Add previous answers as additional context
        if memory:
            additional_context = '\n\n'.join(memory)
            full_context = f"{context}\n\n{additional_context}"
        else:
            full_context = context

        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": full_context},
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

        return f"Sub-question:  {subquestion}\nAnswer:  {response.final_answer}"

    def _generate_answer(self, original_question: str, sub_questions: List[str],
                         retrieve_k: int) -> Dict:
        """Generate final answer from subquestions. 

        Args:
            original_question: The original question.
            sub_questions: List of decomposed subquestions.
            retrieve_k: Number of documents to retrieve.

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys.
        """
        memory = []

        # Answer each subquestion sequentially
        for i, subquestion in enumerate(sub_questions):
            answer_str = self._answer_subquestion(
                subquestion,
                memory[: i],  # Pass only previous answers
                retrieve_k
            )
            memory.append(answer_str)
            logger.debug(f"Answered subquestion {i + 1}/{len(sub_questions)}")

        # Retrieve context for original question
        query_embedding = self.embedding_pipeline.create_embedding(original_question)
        documents = self._retrieve_documents(query_embedding, retrieve_k)

        # Combine all context
        context = '\n\n'.join(memory) + '\n\n' + '\n\n'.join(documents)

        # Generate final answer
        messages = [
            {"role": "system", "content": self.answering_prompt},
            {"role": "user", "content": context},
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
            'knowledge': context
        }

    def generate_answer(self, query: str, retrieve_k: int = None) -> Dict:
        """Generate answer using question decomposition strategy.

        Args:
            query: The user's question.
            retrieve_k: Number of documents to retrieve (uses config default if not specified).

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys. 
        """
        retrieve_k = retrieve_k or self.config.retrieval.retrieve_k

        # Decompose the question
        sub_queries = self._decompose_question(query)

        # Generate and return answer
        return self._generate_answer(query, sub_queries, retrieve_k)