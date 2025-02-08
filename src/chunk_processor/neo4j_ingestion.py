from neo4j import GraphDatabase
import torch
from tools.embedding import EmbeddingPipeline


class DocumentGraphProcessor:
    """
        A class for processing and ingesting text data into a Neo4j graph database.

        Attributes:
            neo4j_url (str): The URL of the Neo4j database.
            neo4j_user (str): The username for the Neo4j database.
            neo4j_password (str): The password for the Neo4j database.
            driver (GraphDatabase.Driver): A driver object for connecting to the Neo4j database.
            embed_pipeline (EmbeddingPipeline): An instance of the EmbeddingPipeline class for generating
                embeddings for text data.
            graph_id (int): The ID of the graph in the Neo4j database.
        """
    def __init__(self, neo4j_url: str, neo4j_user: str, neo4j_password: str,
                 model_name_embedding: str = "BAAI/bge-large-en-v1.5",
                 graph_id: int = 0):
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password))
        self.embed_pipeline = EmbeddingPipeline(model_name_embedding)
        self.graph_id = graph_id

    def close(self):
        if self.driver:
            self.driver.close()

    def create_file_node(self, session, file_title, file_summary):
        query = """
        MERGE (f:File {title: $file_title, summary:$summary, graph_id: $id})
        RETURN f
        """
        session.run(query, file_title=file_title, summary=file_summary, id=self.graph_id)

    def create_chunk_node(self, session, chunk, index):
        query = """
        MERGE (c:Chunk {index: $index, text: $text, embedding: $embedding, graph_id: $id})
        RETURN c
        """
        with torch.no_grad():
            embedding = self.embed_pipeline.create_embedding(chunk)
        session.run(query, index=index, text=chunk, embedding=embedding, id=self.graph_id)

    def create_relationship_between_chunks(self, session, index1, index2):
        query = """
        MATCH (c1:Chunk {index: $index1, graph_id: $id}), (c2:Chunk {index: $index2, graph_id: $id})
        MERGE (c1)-[:NEXT {graph_id: $id}]->(c2)
        """
        session.run(query, index1=index1, index2=index2, id=self.graph_id)

    def create_relationship_file_to_chunk(self, session, index1, index2):
        query = """
        MATCH (c1:File {name: $index1, graph_id: $id}), (c2:Chunk {index: $index2, graph_id: $id})
        MERGE (c1)-[:CONTAINS {graph_id: $id}]->(c2)
        """
        session.run(query, index1=index1, index2=index2, id=self.graph_id)

    def process_chunks(self, dataset_reader):

        previous_index = None
        index = 1
        with self.driver.session() as session:
            for title, chunk, summary in dataset_reader.read_files():

                if chunk.strip():

                    self.create_chunk_node(session, chunk, index)

                    self.create_file_node(session, title, summary)

                    self.create_relationship_file_to_chunk(session, title, index)

                    if previous_index is None or title != previous_title:
                        previous_index = None

                    if previous_index is not None:
                        self.create_relationship_between_chunks(session, previous_index, index)

                    previous_index = index
                    previous_title = title
                    index += 1


class BasicChunkProcessor:
    """
      A simplified version of `GraphChunkProcessor` that only creates nodes for each chunk of text data.

      Attributes:
          neo4j_url (str): The URL of the Neo4j database.
          neo4j_user (str): The username for the Neo4j database.
          neo4j_password (str): The password for the Neo4j database.
          driver (GraphDatabase.Driver): A driver object for connecting to the Neo4j database.
          embed_pipeline (EmbeddingPipeline): An instance of the EmbeddingPipeline class for generating
              embeddings for text data.
          graph_id (int): The ID of the graph in the Neo4j database.
      """
    def __init__(self, neo4j_url: str, neo4j_user: str, neo4j_password: str,
                 model_name_embedding: str = "BAAI/bge-large-en-v1.5",
                 graph_id: int = 0):
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_password))
        self.embed_pipeline = EmbeddingPipeline(model_name_embedding)
        self.graph_id = graph_id

    def close(self):
        if self.driver:
            self.driver.close()

    def create_chunk_node(self, session, chunk, file_title):
        query = """
        MERGE (c:Chunk {title:$title ,text: $text, embedding: $embedding, graph_id: $id})
        RETURN c
        """
        with torch.no_grad():
            embedding = self.embed_pipeline.create_embedding(chunk)
        session.run(query, title=file_title, text=chunk, embedding=embedding, id=self.graph_id)

    def process_chunks(self, dataset_reader):
        with self.driver.session() as session:
            for title, chunk in dataset_reader.read_files():
                if chunk.strip():
                    self.create_chunk_node(session, chunk, title)
