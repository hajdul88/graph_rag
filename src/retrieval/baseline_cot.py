from .base import RAGAgent
from tools.llm_output import ModelResponse
from neo4j import GraphDatabase
from tools.embedding import EmbeddingPipeline
from ollama import Client
from typing import List, Dict
import torch
import json
import time

reasoning_schema = {
    "type": "object",
    "properties": {
        "provided_context": {
            "type": "string",
        },
        "answer_possible": {
            "type": "boolean",
        },
        "final_answer": {
            "type": "string",
        },
        "additional_question": {
            "type": "string",
        }
    },
    "required": ["provided_context", "answer_possible", "final_answer", "additional_question"]
}


class BaselineAgentCoT(RAGAgent):

    def __init__(self,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 answering_prompt_loc: str = "",
                 reasoning_prompt_loc: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 ollama_url: str = 'http://host.docker.internal:11434',
                 reasoning: bool = True,
                 max_reasoning_steps: int = 3
                 ) -> None:

        self.model_name = model_name
        self.client = Client(host=ollama_url)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = EmbeddingPipeline(embedding_model)
        with open(answering_prompt_loc, 'r') as file:
            self.answering_prompt = file.read()
        with open(reasoning_prompt_loc, 'r') as file:
            self.reasoning_prompt = file.read()
        self.reasoning = reasoning
        self.max_reasoning_steps = max_reasoning_steps

    def _retrieve_documents(self, query_embedding: torch.Tensor, k: int = 5) -> List[str]:
        """
        Retrieves the most relevant document chunks from Neo4j based on vector similarity.

        Uses vector search to find document chunks in the database that are most similar
        to the provided query embedding.

        Args:
            query_embedding: A tensor containing the embedding vector of the query.
            k: The number of top documents to retrieve. Defaults to 5.
        Returns:
            A list of text strings representing the retrieved document chunks.
        """
        with self.driver.session() as session:
            query_result = session.run("""
                            // Match chunks, calculate cosine similarity, and return top K results
                            WITH $query_embedding AS query_embedding 
                            CALL db.index.vector.queryNodes('textEmbedding', 500, query_embedding) 
                            YIELD node AS c, score AS similarity
                            RETURN c.text AS text, similarity
                            ORDER BY similarity DESC
                            LIMIT $top_k
                            """, query_embedding=query_embedding, top_k=k)
            return [d['text'] for d in query_result.data()]

    def _generate_answer(self, query: str, documents: List[str], retrieve_k: int) -> Dict:
        """
        Generates an answer to the query using chain-of-thought reasoning over documents.

        Implements an iterative reasoning process that can perform multiple retrieval steps
        if the initial context is insufficient to answer the query. For each step, the model
        evaluates if an answer is possible with the current context, and if not, formulates
        a follow-up question for additional retrieval.

        Args:
            query: The user's question.
            documents: Initial list of retrieved document chunks.
            retrieve_k: Number of documents to retrieve in follow-up retrieval steps.

        Returns:
            A dictionary containing the final answer, reasoning process, and knowledge context.
    """
        context = '\n\n'.join(documents)
        if self.reasoning:
            for _ in range(self.max_reasoning_steps):
                # Prepare messages for the model
                messages = [{"role": "system", "content": self.reasoning_prompt},
                            {"role": "user", "content": query},
                           {"role": "user", "content": context}]
                # Generate and return response
                reasoning_result = self.client.chat(model=self.model_name, messages=messages, stream=False,
                                                    keep_alive=0,
                                                    format=reasoning_schema, options={"temperature": 0.1})
                reasoning_response = json.loads(reasoning_result['message']['content'])
                if reasoning_response['answer_possible']:
                    break
                summarized_context = reasoning_response['provided_context']
                q = reasoning_response['additional_question']
                query_embedding = self.embedding_pipeline.create_embedding(q)
                documents = self._retrieve_documents(query_embedding, retrieve_k)
                context = '\n\n'.join(documents) + "\n\n" + summarized_context
        messages = [{"role": "system", "content": self.answering_prompt}, {"role": "user", "content": context},
                        {"role": "user", "content": query}]
        while True:
            try:
                result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                    format=ModelResponse.model_json_schema(), options={"temperature": 0.1})
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        response = ModelResponse.validate(json.loads(result['message']['content']))
        return {'answer': response.final_answer, 'reasoning': response.reasoning, 'knowledge': context}

    def generate_answer(self, query: str, retrieve_k: int = 5) -> Dict:
        """Main entry point for generating answers to user queries.

         Embeds the query, retrieves relevant documents, and generates an answer using the chain-of-thought reasoning process.

        Args:
            query: The user's question.
            retrieve_k: Number of documents to retrieve. Defaults to 5.

        Returns:
            A dictionary containing the final answer, reasoning process, and knowledge context.
        """
        # Generate query embedding and retrieve relevant documents
        query_embedding = self.embedding_pipeline.create_embedding(query)
        documents = self._retrieve_documents(query_embedding, retrieve_k)

        # Generate and return answer
        return self._generate_answer(query, documents, retrieve_k)
