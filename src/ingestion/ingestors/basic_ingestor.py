from ingestion.chunking.text_chunking import chunk_documents
from tools.embedding import EmbeddingPipeline
from .base import BaseIngestor
from neo4j import GraphDatabase
from typing import List


class BasicIngestor(BaseIngestor):
    def __init__(self, neo4j_url: str, neo4j_user: str, neo4j_password: str,
                 model_name_embedding: str = "BAAI/bge-large-en-v1.5",
                 chunking_method: str = "word_based", chunk_size: int = 800,
                 overlap_size: int = 200):
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        self.embed_pipeline = EmbeddingPipeline(model_name_embedding)

    def create_chunk_node(self, session, chunk, index):
        query = """
        MERGE (d:Document {doc_id:$doc_id ,text: $text, embedding: $embedding})
        RETURN d
        """
        embedding = self.embed_pipeline.create_embedding(chunk)
        session.run(query, doc_id=index, text=chunk, embedding=embedding)

    def ingest(self, corpus_list: List[str]):
        with self.driver.session() as session:
            for i, chunk in chunk_documents(corpus_list, self.chunking_method, self.chunk_size, self.overlap_size):
                if chunk.strip():
                    self.create_chunk_node(session, chunk, i)
        self.driver.close()
