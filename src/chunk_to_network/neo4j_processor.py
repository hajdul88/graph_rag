from neo4j import GraphDatabase
import torch
from transformers import AutoTokenizer, pipeline


class ChunkProcessor:
    """
        Processes text chunks and manages Neo4j graph database operations.

        Creates and manages:
        - File nodes
        - Chunk nodes with embeddings
        - Relationships between chunks
        - Relationships between files and chunks

        Attributes:
            driver: Neo4j database connection
            embed_pipeline: Transformer pipeline for embeddings
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_name):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_pipeline = pipeline(
            "feature-extraction",
            model=model_name,
            tokenizer=self._tokenizer,
            device="cpu"
        )
    def close(self):
        if self.driver:
            self.driver.close()

    def create_file_node(self, session, file_name):
        query = """
        MERGE (f:File {name: $file_name})
        RETURN f
        """
        if file_name.endswith(".txt"):
            file_name = file_name[0:-4]
        session.run(query, file_name=file_name)

    def create_chunk_node(self, session, chunk, index):
        query = """
        MERGE (c:Chunk {index: $index, text: $text, embedding: $embedding})
        RETURN c
        """
        with torch.no_grad():
            embedding = self.embed_pipeline(chunk, truncation=True)[0][0]
        session.run(query, index=index, text=chunk, embedding=embedding)

    def create_relationship_between_chunks(self, session, index1, index2):
        query = """
        MATCH (c1:Chunk {index: $index1}), (c2:Chunk {index: $index2})
        MERGE (c1)-[:NEXT]->(c2)
        """
        session.run(query, index1=index1, index2=index2)

    def create_relationship_file_to_chunk(self, session, index1, index2):
        query = """
        MATCH (c1:File {name: $index1}), (c2:Chunk {index: $index2})
        MERGE (c1)-[:CONTAINS]->(c2)
        """
        if index1.endswith(".txt"):
            index1 = index1[0:-4]
        session.run(query, index1=index1, index2=index2)

    async def process_chunks(self, directory_reader):

        previous_index = None
        index = 1
        with self.driver.session() as session:
            async for file_name, chunk in directory_reader.read_files():
                if chunk.strip():

                    self.create_chunk_node(session, chunk, index)

                    self.create_file_node(session, file_name)

                    self.create_relationship_file_to_chunk(session, file_name, index)

                    if previous_index is None or file_name != previous_file_name:
                        previous_index = None

                    if previous_index is not None:
                        self.create_relationship_between_chunks(session, previous_index, index)

                    previous_index = index
                    previous_file_name = file_name
                    index += 1
        print("Chunk processing complete.")

