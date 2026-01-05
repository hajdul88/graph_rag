from .base import RAGAgent
from tools.llm_output import ModelResponse
from neo4j import GraphDatabase
from tools.embedding import EmbeddingPipeline
from ollama import Client
from typing import List, Dict
import torch
import json

reasoning_schema = {
    "type": "object",
    "properties": {
        "original_question": {
            "type": "string"
        },
        "subquestions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "question": {
                        "type": "string",
                        "minLength": 1
                    }
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

    def __init__(self,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 answering_prompt_loc: str = "",
                 reasoning_prompt_loc: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 ollama_url: str = 'http://host.docker.internal:11434',
                 ) -> None:

        self.model_name = model_name
        self.client = Client(host=ollama_url)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = EmbeddingPipeline(embedding_model)
        with open(answering_prompt_loc, 'r') as file:
            self.answering_prompt = file.read()
        with open(reasoning_prompt_loc, 'r') as file:
            self.reasoning_prompt = file.read()

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

    def _generate_answer(self, original_question: str, sub_questions: List[str], retrieve_k: int) -> Dict:
        """
            TODO: Write docsting
        """
        memory = []
        if len(sub_questions) != 0:
            for i, q in enumerate(sub_questions):
                if i == 0:
                    query_embedding = self.embedding_pipeline.create_embedding(q)
                    documents = self._retrieve_documents(query_embedding, retrieve_k)
                    context = '\n\n'.join(documents)
                    messages = [{"role": "system", "content": self.answering_prompt},
                                {"role": "user", "content": context},
                                {"role": "user", "content": q}]
                    temp_result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                                   format=ModelResponse.model_json_schema())
                    temp_response = ModelResponse.validate(json.loads(temp_result['message']['content']))
                    q_a = f"Sub-question: {q} \n Answer: {temp_response.final_answer}"
                    memory.append(q_a)
                else:
                    extended_q = q + "\n\n" + '\n\n'.join(memory[:i])
                    query_embedding = self.embedding_pipeline.create_embedding(extended_q)
                    documents = self._retrieve_documents(query_embedding, retrieve_k)
                    context = '\n\n'.join(documents)
                    additional_context = '\n\n'.join(memory[:i])
                    messages = [{"role": "system", "content": self.answering_prompt},
                                {"role": "user", "content": context + "\n\n" + additional_context},
                                {"role": "user", "content": q}]
                    temp_result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                                   format=ModelResponse.model_json_schema())
                    temp_response = ModelResponse.validate(json.loads(temp_result['message']['content']))
                    q_a = f"Sub-question: {q} \n Answer: {temp_response.final_answer}"
                    memory.append(q_a)
        query_embedding = self.embedding_pipeline.create_embedding(original_question)
        documents = self._retrieve_documents(query_embedding, retrieve_k)
        context = '\n\n'.join(memory) + '\n\n' + '\n\n'.join(documents)
        messages = [{"role": "system", "content": self.answering_prompt},
                    {"role": "user", "content": context},
                    {"role": "user", "content": original_question}]
        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                  format=ModelResponse.model_json_schema())
        response = ModelResponse.validate(json.loads(result['message']['content']))
        return {'answer': response.final_answer, 'reasoning': response.reasoning, 'knowledge': context}

    def generate_answer(self, query: str, retrieve_k: int = 5) -> Dict:
        """
        TODO: Write docsting
        """
        messages = [{"role": "system", "content": self.reasoning_prompt},
                    {"role": "user", "content": query}]
        reasoning_result = self.client.chat(model=self.model_name, messages=messages, stream=False,
                                            keep_alive=0,
                                            format=reasoning_schema, options={"temperature": 0.1})
        reasoning_response = json.loads(reasoning_result['message']['content'])
        sub_queries = [q['question'] for q in reasoning_response['subquestions']]

        # Generate and return answer
        return self._generate_answer(query, sub_queries, retrieve_k)
