from transformers import AutoTokenizer, pipeline
import torch
import numpy as np
from neo4j import GraphDatabase


class DummyRetriever:
    """
        Retrieves relevant documents using embedding similarity in Neo4j.

        Attributes:
            tokenizer: Transformer tokenizer for text processing
            embed_pipeline: Pipeline for generating embeddings
            driver: Neo4j database connection
        """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_pipeline = pipeline(
            "feature-extraction",
            model=model_name,
            tokenizer=self.tokenizer,
            device="cpu"
        )
        self.driver = GraphDatabase.driver(neo4j_uri,
                                           auth=(neo4j_user, neo4j_password))  # Replace with your credentials

    def get_embeddings(self, text):
        """
                Generates embeddings for input text using the transformer model.

                Args:
                    text: Input text to embed

                Returns:
                    numpy.ndarray: Text embedding vector
        """
        with torch.no_grad():
            embedding_ = self.embed_pipeline(text, truncation=True)[0][0]
        return embedding_

    def retrieve(self, query, top_k=5):
        """
                Retrieves most similar documents to input query.

                Process:
                1. Generates query embedding
                2. Executes Neo4j similarity search
                3. Returns top-k most similar documents

                Args:
                    query: Search query text
                    top_k: Number of results to return (default: 5)

                Returns:
                    List of dictionaries containing file names and similarity scores
                """
        query_embedding = self.get_embeddings(query)

        # Convert query embedding to a list for Cypher
        query_embedding_list = query_embedding

        with self.driver.session() as session:
            result = session.run("""
                // Match chunks, calculate cosine similarity, and return top K results
                WITH $query_embedding AS query_embedding
                    MATCH (f:File)-[:CONTAINS]->(c:Chunk)
                    WITH f.name AS file_name, gds.similarity.cosine(c.embedding, query_embedding) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                RETURN file_name, similarity
            """, query_embedding=query_embedding_list, top_k=top_k)
            # Collect and return file names with their similarity scores
            return result.data()
