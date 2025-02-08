from .base import RAGAgent
from neo4j import GraphDatabase
from transformers import AutoTokenizer, pipeline
from ollama import Client
from typing import List, Dict, Any
import torch
from tools.embedding import RerankingPipeline


class BaselineAgent(RAGAgent):

    def __init__(self,
                 graph_id: int = None,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 reranking_model: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self.graph_id = graph_id
        self.client = Client(host='http://host.docker.internal:11434')
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = pipeline(
            "feature-extraction",
            model=embedding_model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.reranker = RerankingPipeline(model_name=reranking_model)

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
                            WITH $query_embedding AS query_embedding, $id AS graph_id
                            CALL db.index.vector.queryNodes('textEmbedding', 500, query_embedding) 
                            YIELD node AS c, score AS similarity
                            WHERE c.graph_id = graph_id
                            RETURN c.title AS title, c.text AS text, similarity
                            ORDER BY similarity DESC
                            LIMIT $top_k
                            """, query_embedding=query_embedding, top_k=k, id=self.graph_id)
            return query_result.data()

    def _rerank_documents(self, query: str, documents: List[Dict[str, str]]):
        doc_texts = [d['text'] for d in documents]
        indices = self.reranker.rerank(query, doc_texts)
        doc_titles = [doc['title'] for doc in documents]
        doc_titles = [doc_titles[i] for i in indices]
        return doc_titles

    def retrieve_relevant_docs(self, query: str = "", k: int = 5) -> List[str]:
        """ Retrieve k most relevant documents according to
            the largest similarity of the contained chunk

                Args:
                    query (str): User query
                    k (int): number of documents

                Returns:
                    List[str]: document list
        """
        query_embedding = self.embedding_pipeline(query)[0][0]
        documents = self._retrieve_documents(query_embedding, k)
        doc_list = self._rerank_documents(query, documents)
        return doc_list

    def _generate_answer(self, query: str, documents: List[Dict[str, str]]) -> str:
        """Generates answer based on query and retrieved documents.

        Args:
            query (str): User query
            documents (List[Dict[str, str]]): Retrieved documents with title and summary

        Returns:
            str: Generated answer
        """
        # Define system template for response generation
        system_template = ("Below is a question followed by some context from different sources. Please answer the "
                           "question based on the context."
                           "The answer to the question is a word or entity. If the provided information is "
                           "insufficient to answer the question,"
                           "respond 'Insufficient Information'. Answer directly without explanation.")

        # Prepare messages for the model
        messages = [{"role": "system", "content": system_template}]
        for doc in documents:
            doc_input = f"Title: {doc['title']} Text: {doc['text']}"
            messages.append({"role": "user", "content": doc_input})
        messages.append({"role": "user", "content": query})

        # Generate and return response
        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0)
        return result['message']['content']

    def generate_answer(self, query: str, retrieve_k: int = 5) -> str:
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


class BasicMultiHopAgent(RAGAgent):
    """Agent for performing multi-hop reasoning using retrieved documents.

    Args:
        model_name (str, optional): Name of the language model.
        neo4j_url (str, optional): Neo4j database URL. Defaults to "".
        neo4j_username (str, optional): Neo4j username. Defaults to "".
        neo4j_pw (str, optional): Neo4j password. Defaults to "".
        embedding_model (str, optional): Name of embedding model. Defaults to "sentence-transformers/all-mpnet-base-v2".
    """

    def __init__(self,
                 graph_id: int = None,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5") -> None:
        self.model_name = model_name
        self.graph_id = graph_id
        self.client = Client(host='http://host.docker.internal:11434')
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = pipeline(
            "feature-extraction",
            model=embedding_model,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _generate_answer(self, query: str, documents: List[Dict[str, str]]) -> str:
        """Generates answer based on query and retrieved documents.

        Args:
            query (str): User query
            documents (List[Dict[str, str]]): Retrieved documents with title and summary

        Returns:
            str: Generated answer
        """
        # Define system template for response generation
        system_template = ("Below is a question followed by some context from different sources. Please answer the "
                           "question based on the context."
                           "The answer to the question is a word or entity. If the provided information is "
                           "insufficient to answer the question,"
                           "respond 'Insufficient Information'. Answer directly without explanation.")

        # Prepare messages for the model
        messages = [{"role": "system", "content": system_template}]
        for doc in documents:
            doc_input = f"Title: {doc['title']} Text: {doc['summary']}"
            messages.append({"role": "user", "content": doc_input})
        messages.append({"role": "user", "content": query})

        # Generate and return response
        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0)
        return result['message']['content']

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
                            MATCH (c {graph_id:$id})<-[:CONTAINS {graph_id:$id}]-(n:File {graph_id:$id})
                            WITH n.title AS title, n.summary AS summary, MAX(similarity) AS max_similarity
                            ORDER BY max_similarity DESC
                            RETURN title, summary, max_similarity
                            LIMIT $top_k
                            """, query_embedding=query_embedding, top_k=k, id=self.graph_id)
            return query_result.data()

    def generate_answer(self, query: str, retrieve_k: int = 5) -> str:
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

    def retrieve_relevant_docs(self, query: str = "", k: int = 5) -> List[str]:
        """ Retrieve k most relevant documents according to
            the largest similarity of the contained chunk

                Args:
                    query (str): User query
                    k (int): number of documents

                Returns:
                    List[str]: document list
        """
        query_embedding = self.embedding_pipeline(query)[0][0]
        documents = self._retrieve_documents(query_embedding, k)
        doc_list = [doc['title'] for doc in documents]
        return doc_list
