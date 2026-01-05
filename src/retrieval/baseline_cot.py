import json
import time
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
        "provided_context": {"type": "string"},
        "answer_possible": {"type": "boolean"},
        "final_answer": {"type": "string"},
        "additional_question": {"type": "string"}
    },
    "required": ["provided_context", "answer_possible", "final_answer", "additional_question"]
}


class BaselineAgentCoT(RAGAgent):
    """Baseline agent using chain-of-thought reasoning with iterative retrieval. 

    This agent performs iterative reasoning over retrieved documents,
    asking follow-up questions when the initial context is insufficient.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the BaselineAgentCoT. 

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

    def _retrieve_documents(self, query_embedding: torch.Tensor, k: int = 5) -> List[str]:
        """Retrieve the most relevant document chunks from Neo4j. 

        Uses vector similarity search to find document chunks most similar
        to the provided query embedding.

        Args:
            query_embedding: Tensor containing the embedding vector of the query.
            k: Number of top documents to retrieve.  Defaults to 5.

        Returns:
            A list of text strings representing the retrieved document chunks.
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

    def _apply_reasoning_steps(self, query: str, context: str, retrieve_k: int) -> tuple:
        """Apply iterative reasoning to refine context.

        Performs multiple reasoning steps where the model evaluates if the current
        context is sufficient to answer the query, and if not, formulates follow-up
        questions for additional retrieval.

        Args:
            query: The user's question. 
            context: Initial retrieved context.
            retrieve_k: Number of documents to retrieve in follow-up steps.

        Returns:
            Tuple of (final_context, reasoning_response).
        """
        reasoning_response = None

        for step in range(self.config.reasoning.max_steps):
            messages = [
                {"role": "system", "content": self.reasoning_prompt},
                {"role": "user", "content": query},
                {"role": "user", "content": context}
            ]

            try:
                result = self.client.chat(
                    model=self.config.model_name,
                    messages=messages,
                    stream=False,
                    keep_alive=0,
                    format=reasoning_schema,
                    options={"temperature": self.config.reasoning.temperature}
                )
                reasoning_response = json.loads(result['message']['content'])
            except Exception as e:
                logger.error(f"Reasoning step {step} failed: {e}")
                continue

            # If answer is possible with current context, stop reasoning
            if reasoning_response['answer_possible']:
                logger.debug(f"Answer possible after {step + 1} reasoning steps")
                break

            # Refine context with follow-up question
            summarized_context = reasoning_response['provided_context']
            follow_up_question = reasoning_response['additional_question']

            logger.debug(f"Follow-up question {step + 1}: {follow_up_question}")

            query_embedding = self.embedding_pipeline.create_embedding(follow_up_question)
            additional_documents = self._retrieve_documents(query_embedding, retrieve_k)
            additional_context = '\n\n'.join(additional_documents)

            context = f"{additional_context}\n\n{summarized_context}"

        return context, reasoning_response

    def _generate_answer(self, query: str, documents: List[str], retrieve_k: int) -> Dict:
        """Generate an answer to the query using chain-of-thought reasoning.

        Implements an iterative reasoning process with optional multi-step retrieval
        followed by final answer generation.

        Args:
            query: The user's question. 
            documents: Initial list of retrieved document chunks.
            retrieve_k: Number of documents to retrieve in follow-up steps.

        Returns:
            A dictionary containing 'answer', 'reasoning', and 'knowledge' keys.
        """
        context = '\n\n'.join(documents)

        # Apply reasoning steps if enabled
        if self.config.reasoning.enabled:
            context, _ = self._apply_reasoning_steps(query, context, retrieve_k)

        # Generate final answer
        answer_response = self._llm_final_answer(query, context)
        return answer_response

    def _llm_final_answer(self, query: str, context: str) -> Dict:
        """Query LLM for final answer with retry logic.

        Args:
            query: The user's question.
            context: The context to use for answer generation.

        Returns:
            Dictionary with 'answer', 'reasoning', and 'knowledge' keys. 
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
                    logger.error(f"Failed to generate answer after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(retry_delay)

    def generate_answer(self, query: str, retrieve_k: int = None) -> Dict:
        """Generate answer to user query. 

        Main entry point that embeds the query, retrieves relevant documents,
        and generates an answer using chain-of-thought reasoning. 

        Args:
            query: The user's question.
            retrieve_k: Number of documents to retrieve.  Uses config default if not specified.

        Returns:
            Dictionary containing 'answer', 'reasoning', and 'knowledge' keys. 
        """
        retrieve_k = retrieve_k or self.config.retrieval.retrieve_k

        query_embedding = self.embedding_pipeline.create_embedding(query)
        documents = self._retrieve_documents(query_embedding, retrieve_k)

        return self._generate_answer(query, documents, retrieve_k)