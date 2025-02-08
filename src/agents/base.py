from abc import ABC


class RAGAgent(ABC):
    """
    Abstract base class for RAG agents performing document retrieval and answer generation.

    Attributes:
        model_name (str): Name of the language model.
        neo4j_url (str): Neo4j database URL.
        neo4j_username (str): Neo4j username.
        neo4j_pw (str): Neo4j password.
        embedding_model (str): Name of embedding model.
        graph_id (int, optional): Identifier for the graph database context.
    """

    def generate_answer(self, query: str, retrieve_k: int) -> str:
        """
        Main method for generating answers:
        1. Create query embedding.
        2. Retrieve relevant documents.
        3. Generate and return an answer based on retrieved content.
        """
        pass

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> list:
        """
        Main method for retrieving relevant documents:
        1. Create query embedding.
        2. Query the database.
        3. Return a list of document titles.
        """
        pass