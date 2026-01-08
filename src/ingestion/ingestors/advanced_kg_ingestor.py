from ingestion.chunking.text_chunking import chunk_documents
from tools.embedding import EmbeddingPipeline
from .base import BaseIngestor
from neo4j import GraphDatabase
from typing import List
from knowledge_graphs.advanced_construction_pipeline import AdvancedKGConstructor


class AdvancedKGIngestor(BaseIngestor):
    def __init__(self, neo4j_url: str, neo4j_user: str, neo4j_password: str,
                 template_re_loc: str,
                 template_ner_loc: str,
                 ner_model: str = "qwen2.5:7b",
                 re_model: str = "qwen2.5:7b",
                 model_name_embedding: str = "BAAI/bge-large-en-v1.5",
                 chunking_method: str = "word_based",
                 chunk_size: int = 500,
                 overlap_size: int = 200,
                 llm_endpoint_url: str = ""):
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        self.embed_pipeline = EmbeddingPipeline(model_name_embedding)
        self.kg_pipeline = AdvancedKGConstructor(template_re_loc=template_re_loc, template_ner_loc=template_ner_loc,
                                                 ner_model=ner_model, re_model=re_model,
                                                 embedding_pipeline=self.embed_pipeline,
                                                 llm_endpoint_url=llm_endpoint_url)

    def create_entity_description(self, session, e_description, e_embedding, e_name):
        query = """
            MERGE (d:Description {text: $e_description, embedding: $e_embedding})
            WITH d
            MATCH (e:Entity {name: $e_name})
            MERGE (d)-[r:DESCRIBES]->(e)
            RETURN r
            """
        session.run(query, e_description=e_description, e_embedding=e_embedding, e_name=e_name)

    def create_entity_node(self, session, e_name, e_type, e_aliases):
        e_aliases.append(e_name)
        query = """
        OPTIONAL MATCH (e:Entity)
        WHERE e.name = $e_name 
        OR $e_name IN e.aliases 
        OR any(alias IN $e_aliases WHERE alias = e.name)
        WITH e
        CALL apoc.do.when(
            e IS NULL,
            'CREATE (newEntity:Entity {name: $e_name, e_type: $e_type, aliases: $e_aliases}) RETURN newEntity AS resultEntity',
            'SET entity.aliases = entity.aliases + [x IN $e_aliases WHERE NOT x IN entity.aliases],
            entity.e_type = $e_type
            RETURN entity AS resultEntity',
            {entity: e, e_name: $e_name, e_type: $e_type, e_aliases: $e_aliases}
        ) YIELD value
        RETURN value.resultEntity.name AS name
        """
        result = session.run(query, e_name=e_name, e_type=e_type, e_aliases=e_aliases)
        for record in result:
            return record['name']

    def create_relationship_entity_entity(self, session, h_name, t_name, rel_name, rel_embedding):
        query = """
            MATCH (e1:Entity {name: $h_name}), (e2:Entity {name: $t_name})
            MERGE (e1)-[r:RELATED_TO {name: $rel_name}]->(e2)
            ON CREATE SET r.embedding = $rel_embedding
            """
        session.run(query, h_name=h_name, t_name=t_name, rel_name=rel_name, rel_embedding=rel_embedding)

    def create_chunk_node(self, session, chunk, index):
        query = """
        MERGE (d:Document {doc_id:$doc_id ,text: $text, embedding: $embedding})
        RETURN d
        """
        embedding = self.embed_pipeline.create_embedding(chunk)
        session.run(query, doc_id=index, text=chunk, embedding=embedding)

    def connect_chunk_entity(self, session, chunk_id, entity_name):
        query = """
        MATCH (e:Entity {name: $e_name}), (d:Document {doc_id:$doc_id})
        MERGE (d)-[r:DESCRIBES]->(e)
        RETURN r
        """
        session.run(query, e_name=entity_name, doc_id=chunk_id)

    def ingest(self, corpus_list: List[str]):
        with self.driver.session() as s:
            for i, chunk in chunk_documents(corpus_list, self.chunking_method, self.chunk_size, self.overlap_size):
                entities, relationships, relation_embeddings = self.kg_pipeline.extract_entities_and_relationships(
                    chunk)
                self.create_chunk_node(session=s, chunk=chunk, index=i)
                for e in entities:
                    retrieved_name = self.create_entity_node(session=s, e_name=e['name'], e_type=e['type'],
                                                             e_aliases=e['aliases'])
                    self.create_entity_description(session=s, e_description=e['entity_information'],
                                                   e_embedding=e['embedding'], e_name=retrieved_name)
                    for r, em in zip(relationships, relation_embeddings):
                        self.create_relationship_entity_entity(session=s, h_name=r[0], t_name=r[2], rel_name=r[1],
                                                               rel_embedding=em)
                    self.connect_chunk_entity(session=s, chunk_id=i, entity_name=e['name'])
        self.driver.close()
