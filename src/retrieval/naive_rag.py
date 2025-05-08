from .base import RAGAgent
from tools.llm_output import ModelResponse
from neo4j import GraphDatabase
from transformers import pipeline
from ollama import Client
from typing import List, Dict, Any
import torch
import json


class BaselineAgent(RAGAgent):

    def __init__(self,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 ollama_url: str = 'http://host.docker.internal:11434') -> None:
        self.model_name = model_name
        self.client = Client(host=ollama_url)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = pipeline(
            "feature-extraction",
            model=embedding_model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _retrieve_documents(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves relevant documents based on query embedding.

        Args:
            query_embedding (torch.Tensor): Embedding vector of the query
            k (int): Number of files to retrieve

        Returns:
            List[Dict[str, Any]]: Retrieved documents with title, summary, and similarity scores
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
            return query_result.data()

    def _generate_answer(self, query: str, documents: List[Dict[str, str]]) -> Dict:
        """Generates answer based on query and retrieved documents.

        Args:
            query (str): User query
            documents (List[Dict[str, str]]): Retrieved documents with title and summary

        Returns:
            str: Generated answer
        """
        # Define system template for response generation
        system_template = ("Below is a question followed by some context from different sources."
                           "Please answer the question based on the context. The answer to the question could be "
                           "either single word, yes/no or consist of multiple words describing single entity."
                           "If the provided information is insufficient to answer the question,"
                           "respond 'Insufficient Information'. Answer directly without explanation."
                           "Provide answer in JSON object with two string attributes, 'reasoning', which"
                           "provides your detailed reasoning about the answer, and"
                           "'final_answer' where you provide your short final answer without explaining your reasoning."
                           )

        # Prepare messages for the model
        messages = [{"role": "system", "content": system_template}]
        knowledge_paragraphs = []
        for i, doc in enumerate(documents):
            doc_input = f"Paragraph {i} : {doc['text']}"
            knowledge_paragraphs.append(doc_input)
            messages.append({"role": "user", "content": doc_input})
        messages.append({"role": "user", "content": query})

        # Generate and return response
        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                  format=ModelResponse.model_json_schema())
        response = ModelResponse.validate(json.loads(result['message']['content']))
        knowledge = '\n'.join(knowledge_paragraphs)
        return {'answer': response.final_answer, 'reasoning': response.reasoning, 'knowledge': knowledge}

    def generate_answer(self, query: str, retrieve_k: int = 5) -> Dict:
        """Generates answer for given query using retrieved documents.

        Args:
            query (str): User query
            retrieve_k (int): Number of documents to be retrieved
        Returns:
            str: Generated answer based on retrieved documents
        """
        # Generate query embedding and retrieve relevant documents
        query_embedding = self.embedding_pipeline(query)[0][0]
        documents = self._retrieve_documents(query_embedding, retrieve_k)

        # Generate and return answer
        return self._generate_answer(query, documents)
