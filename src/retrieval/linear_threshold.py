from .base import RAGAgent
from tools.embedding import EmbeddingPipeline
from tools.llm_output import ModelResponse
from ollama import Client
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Tuple
import json


class LinearThresholdAgent(RAGAgent):
    """A RAG agent that uses linear threshold model for knowledge graph traversal.

    This agent implements a graph-based retrieval approach using linear threshold
    activation to find relevant entities and relationships in a knowledge graph.

    Args:
        model_name (str, optional): Name of the LLM model to use. Defaults to "hermes3".
        neo4j_url (str, optional): URL for Neo4j database connection. Defaults to "".
        neo4j_username (str, optional): Username for Neo4j authentication. Defaults to "".
        neo4j_pw (str, optional): Password for Neo4j authentication. Defaults to "".
        embedding_model (str, optional): Name of the embedding model. Defaults to "BAAI/bge-large-en-v1.5".
        number_of_iterations (int, optional): Number of iterations for linear threshold. Defaults to 5.
        max_edge_probability (float, optional): Maximum probability for edge weights. Defaults to 1.0.
    """

    def __init__(self,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 number_of_iterations: int = 5,
                 max_edge_probability: float = 1.0,
                 ollama_url:str = 'http://host.docker.internal:11434'):
        self.iterations = number_of_iterations
        self.max_edge_probability = max_edge_probability
        self.model_name = model_name
        self.client = Client(host=ollama_url)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = EmbeddingPipeline(embedding_model)

    def _retrieve_context(self, query_embedding: np.array, k_documents: int) -> List[Dict]:
        """Retrieves relevant context from the knowledge graph based on query embedding.

        Args:
            query_embedding (np.array): The embedding vector of the input query.
            k_documents (int): Number of top documents to consider.

        Returns:
            List[Dict]: List of unique triplets containing head entity, relationship, and tail entity information.
        """
        tail_query = """
                WITH $query_embedding AS queryEmbedding
                MATCH (d:Document)
                WITH d, gds.similarity.cosine(queryEmbedding, d.embedding) AS sim
                ORDER BY sim DESC
                LIMIT $k_docs
                MATCH (d)-[:CONTAINS]->(e:Entity)
                WITH DISTINCT e, $query_embedding AS queryEmbedding
                MATCH (e)-[r:RELATED_TO]->(other:Entity)
                WITH e, other, r, 
                    gds.similarity.cosine(queryEmbedding, r.embedding) AS similarity,
                    1 - gds.similarity.cosine(queryEmbedding, e.embedding) AS head_distance,
                    1 - gds.similarity.cosine(queryEmbedding, other.embedding) AS tail_distance
                ORDER BY similarity DESC
                WITH e, other, head(collect({r: r, similarity: similarity})) AS bestRel, tail_distance, head_distance
                RETURN e.name AS head_entity_name, 
                    bestRel.r.name AS relationship_name, 
                    bestRel.similarity AS similarity, 
                    other.name AS tail_entity_name, 
                    tail_distance AS tail_distance,
                    head_distance AS head_distance
                """

        head_query = """
            WITH $query_embedding AS queryEmbedding
            MATCH (d:Document)
            WITH d, gds.similarity.cosine(queryEmbedding, d.embedding) AS sim
            ORDER BY sim DESC
            LIMIT $k_docs
            MATCH (d)-[:CONTAINS]->(e:Entity)
            WITH DISTINCT e, $query_embedding AS queryEmbedding
            MATCH (e)<-[r:RELATED_TO]-(other:Entity)
            WITH e, other, r, 
                gds.similarity.cosine(queryEmbedding, r.embedding) AS similarity,
                1 - gds.similarity.cosine(queryEmbedding, e.embedding) AS tail_distance,
                1 - gds.similarity.cosine(queryEmbedding, other.embedding) AS head_distance
            ORDER BY similarity DESC
            WITH e, other, head(collect({r: r, similarity: similarity})) AS bestRel, tail_distance, head_distance
            RETURN other.name AS head_entity_name, 
                bestRel.r.name AS relationship_name, 
                bestRel.similarity AS similarity, 
                e.name AS tail_entity_name, 
                tail_distance AS tail_distance,
                head_distance AS head_distance
            """

        with self.driver.session() as session:
            tail_results = session.run(tail_query, query_embedding=query_embedding, k_docs=k_documents)
            head_results = session.run(head_query, query_embedding=query_embedding, k_docs=k_documents)
            # Collecting rows as a list of dictionaries
            unique_triplets = []
            seen = set()
            for record in head_results:
                data = record.data()
                key = (data['head_entity_name'], data['relationship_name'], data['tail_entity_name'])
                seen.add(key)
                unique_triplets.append(data)

            for record in tail_results:
                data = record.data()
                key = (data['head_entity_name'], data['relationship_name'], data['tail_entity_name'])
                if key not in seen:
                    unique_triplets.append(data)
            return unique_triplets

    def _retrieve_activated_entities(self, query_embedding: np.array, k_documents: int, k_entities) -> List[str]:
        """Retrieves initially activated entities based on similarity to query.

        Args:
            query_embedding (np.array): The embedding vector of the input query.
            k_documents (int): Number of top documents to consider.
            k_entities (int): Number of top entities to retrieve.

        Returns:
            List[str]: List of entity names that are initially activated.
        """
        query = """
                    WITH $query_embedding AS queryEmbedding
                    MATCH (d:Document)
                    WITH d, gds.similarity.cosine(queryEmbedding, d.embedding) AS sim
                    ORDER BY sim DESC
                    LIMIT $k_docs
                    MATCH (d)-[:CONTAINS]->(e:Entity)
                    WHERE (e)-[:RELATED_TO]-(:Entity)
                    WITH DISTINCT e, $query_embedding AS queryEmbedding, gds.similarity.cosine($query_embedding, e.embedding) AS sim
                    ORDER BY sim DESC
                    LIMIT $k_ents
                    RETURN e.name AS entity_name
            """
        with self.driver.session() as s:
            result = s.run(query, query_embedding=query_embedding, k_docs=k_documents, k_ents=k_entities)
            return [record.data()['entity_name'] for record in result]

    def _retrieve_relevant_entities(self, entity_list: List[str]):
        """Retrieves detailed information about specified entities.

        Args:
            entity_list (List[str]): List of entity names to retrieve information for.

        Returns:
            List[Tuple]: List of tuples containing entity names and their attributes.
        """
        query = """
        MATCH (n:Entity)
        WHERE n.name IN $entity_names
        WITH DISTINCT n
        RETURN n.name as name, n.attrs as attributes

        """
        with self.driver.session() as s:
            result = s.run(query, entity_names=entity_list)
            return [(record.data()['name'], ' , '.join(record.data()['attributes'])) for record in result]

    def _create_adj_dict(self, arc_list: List[Dict]):
        """Creates adjacency dictionary and threshold values from arc list.

          Args:
              arc_list (List[Dict]): List of dictionaries containing relationship information.

          Returns:
              Tuple[Dict, Dict]: A tuple containing:
                  - adjacency dictionary mapping entities to their connections
                  - threshold dictionary mapping entities to their threshold values
          """
        thresholds = dict()
        adj_dict = dict()

        for j, a in enumerate(arc_list):
            if not a['head_entity_name'] in thresholds:
                thresholds[a['head_entity_name']] = a['head_distance']
                adj_dict[a['head_entity_name']] = []
            if not a['tail_entity_name'] in thresholds:
                thresholds[a['tail_entity_name']] = a['tail_distance']
                adj_dict[a['tail_entity_name']] = []
            adj_dict[a['head_entity_name']].append(
                (a['tail_entity_name'], j, a['similarity'] * self.max_edge_probability))
            adj_dict[a['tail_entity_name']].append(
                (a['head_entity_name'], j, a['similarity'] * self.max_edge_probability))
        return adj_dict, thresholds

    def _linear_threshold(self,
                          adj_dict: Dict[str, List[Tuple]],
                          thresholds: Dict[str, float],
                          initially_activated: List[str]):
        """Implements linear threshold model for entity activation.

        Args:
            adj_dict (Dict[str, List[Tuple]]): Adjacency dictionary of the graph.
            thresholds (Dict[str, float]): Threshold values for each entity.
            initially_activated (List[str]): List of initially activated entity names.

        Returns:
            Tuple[set, set]: A tuple containing:
                - set of relevant triplet indices
                - set of activated entity names
        """
        A = set(initially_activated)  # Nodes activated in the current time step.
        activated = set(initially_activated)  # All nodes activated so far.
        not_activated = set(thresholds.keys()) - activated
        t = 0
        while t < self.iterations and A:
            newA = set()
            for u in not_activated:
                influence = 0
                for arc in adj_dict[u]:
                    target, arc_index, prob = arc
                    if target in activated:
                        influence += prob
                if thresholds[u] <= min(influence, 1):
                    newA.add(u)
            activated |= newA
            not_activated -= newA
            A = newA
        relevant_triplets = {a[1] for e, arcs in adj_dict.items() if e in activated for a in arcs if a[0] in activated}

        return relevant_triplets, activated

    def _generate_answer(self, query: str, knowledge: str) -> str:
        """Generates answer using LLM based on query and knowledge context.

        Args:
            query (str): The input question.
            knowledge (str): Retrieved knowledge context.

        Returns:
            str: Generated answer from the LLM.
        """
        system_template = ("You are provided with the a multi-hop question."
                           "Additionally, you are given relationship statements that highlight"
                           "key connections between entities. Please answer the question based on the information from"
                           "relationship statements; The answer to the question could be "
                           "either single word, yes/no or consist of multiple words describing single entity."
                           "Provide answer in JSON object with two string attributes, 'reasoning', which"
                           "provides your detailed reasoning about the answer, and"
                           "'final_answer' where you provide your short final answer without explaining your reasoning.")

        # Prepare messages for the model
        messages = [{"role": "system", "content": system_template}]
        #print('-----------------------KNOWLEDGE-----------------------------------------------------------')
        #print(knowledge)
        messages.append({"role": "user", "content": knowledge})
        messages.append({"role": "user", "content": query})
        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                  format=ModelResponse.model_json_schema(), options={"temperature": 0.1})
        response = ModelResponse.validate(json.loads(result['message']['content']))
        #print('-----------------------------LLM_REASONING-----------------------------------------------')
        #print(response.reasoning)
        #print('--------------------------------------------------------------------------------------------------')
        return response.final_answer

    def generate_answer(self, query: str, retrieve_k: int = 5, key_entities: int = 5) -> str:
        """Main method to generate answer for input query using the RAG pipeline.

        Args:
            query (str): The input question to answer.
            retrieve_k (int, optional): Number of documents to retrieve. Defaults to 2.
            key_entities (int, optional): Number of key entities to consider. Defaults to 5.

        Returns:
            str: Final answer generated by the model.
        """
        # Generate query embedding and retrieve relevant documents
        query_embedding = self.embedding_pipeline.create_embedding(query)
        retrieved_rels = self._retrieve_context(query_embedding, retrieve_k)
        activated_entities = self._retrieve_activated_entities(query_embedding, retrieve_k, key_entities)
        adj_dict, thresholds = self._create_adj_dict(retrieved_rels)
        relevant_triplets, relevant_entities = self._linear_threshold(adj_dict=adj_dict,
                                                                      thresholds=thresholds,
                                                                      initially_activated=activated_entities
                                                                      )
        entity_data = self._retrieve_relevant_entities(list(relevant_entities))
        entity_context = [f'NAME: {record[0]} ATTRIBUTES: {record[1]}' for record in entity_data]
        triplets_context = [' '.join((retrieved_rels[i]['head_entity_name'], retrieved_rels[i]['relationship_name'],
                                      retrieved_rels[i]['tail_entity_name']))
                            for i in relevant_triplets]
        context = ("# Context entities" + '\n' + '\n'.join(entity_context) +
                   '\n' + '# Context triplets' + '\n' + '\n'.join(triplets_context))

        # Generate and return answer
        return self._generate_answer(query, context)
