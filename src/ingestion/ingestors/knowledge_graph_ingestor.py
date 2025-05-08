from ingestion.chunking.text_chunking import chunk_documents
from tools.embedding import EmbeddingPipeline
from .base import BaseIngestor
from neo4j import GraphDatabase
from typing import List
from knowledge_graphs.construction_pipeline import KnowledgeGraphConstructor


class KnowledgeGraphIngestor(BaseIngestor):
    def __init__(self, neo4j_url: str, neo4j_user: str, neo4j_password: str,
                 template_re_loc: str,
                 template_ner_loc: str,
                 ner_model: str = "qwen2.5:7b",
                 re_model: str = "qwen2.5:7b",
                 model_name_embedding: str = "BAAI/bge-large-en-v1.5",
                 chunking_method: str = "word_based",
                 chunk_size: int = 500,
                 overlap_size: int = 200,
                 ollama_url: str = ""):
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
        self.embed_pipeline = EmbeddingPipeline(model_name_embedding)
        self.kg_pipeline = KnowledgeGraphConstructor(template_re_loc=template_re_loc, template_ner_loc=template_ner_loc,
                                                     ner_model=ner_model, re_model=re_model,
                                                     embedding_pipeline=self.embed_pipeline,
                                                     ollama_url=ollama_url)

    def create_document_node(self, session, text, doc_id):
        d_embedding = self.embed_pipeline.create_embedding(text)
        query = """
        MERGE (d:Document {doc_id:$id, embedding: $d_embedding, text: $text})
        RETURN d
        """
        return session.run(query, d_embedding=d_embedding, text=text, id=doc_id)

    def create_entity_node(self, session, e_name, e_type, e_attributes, e_embedding):
        query = """
        MERGE (e:Entity {name: $e_name})
        ON CREATE SET  e.e_type = $e_type, e.attrs = $e_attributes, e.embedding=$e_embedding
        ON MATCH SET e.attr = e.attr + [x IN $e_attributes WHERE NOT x IN e.attr]
        RETURN e
        """
        return session.run(query, e_name=e_name, e_type=e_type, e_attributes=e_attributes, e_embedding=e_embedding)

    def create_relationship_entity_entity(self, session, h_name, t_name, rel_name, rel_embedding):
        query = """
            MATCH (e1:Entity {name: $h_name}), (e2:Entity {name: $t_name})
            MERGE (e1)-[r:RELATED_TO {name: $rel_name}]->(e2)
            ON CREATE SET r.embedding = $rel_embedding
            """
        return session.run(query, h_name=h_name, t_name=t_name, rel_name=rel_name, rel_embedding=rel_embedding)

    def create_relationship_document_entity(self, session, doc_id, e_name):
        query = """
        MATCH (d:Document {doc_id: $id}), (e:Entity {name: $e_name})
        MERGE (d)-[:CONTAINS]->(e)
        """
        return session.run(query, id=doc_id, e_name=e_name)

    def ingest(self, corpus_list: List[str]):
        all_triplets = []
        all_embeddings = []
        all_entities = []
        entity_doc_map = dict()
        with self.driver.session() as session:
            for i, chunk in chunk_documents(corpus_list, self.chunking_method, self.chunk_size, self.overlap_size):
                if chunk.strip():
                    self.create_document_node(session, chunk, i)
                    triplets, embeddings, entities = self.kg_pipeline.extract_frequent_relations(iterations=1,
                                                                                                 text=chunk,
                                                                                                 freq_tolerance=0.1)
                    all_triplets.extend(triplets)
                    all_embeddings.extend(embeddings)
                    all_entities.extend(entities)

                    for e in entities:
                        if not entity_doc_map.get(e['entity_name']):
                            entity_doc_map[e['entity_name']] = set()
                        entity_doc_map[e['entity_name']].add(i)

            for e in all_entities:
                self.create_entity_node(session=session, e_name=e['entity_name'], e_type=e['entity_type'],
                                        e_attributes=e['properties'], e_embedding=e['embedding'])
                for doc_id in entity_doc_map[e['entity_name']]:
                    self.create_relationship_document_entity(session=session, doc_id=doc_id,
                                                             e_name=e['entity_name'])
            for t, e in zip(all_triplets, all_embeddings):
                self.create_relationship_entity_entity(session=session, h_name=t[0], t_name=t[2], rel_name=t[1],
                                                       rel_embedding=e)
        self.driver.close()
