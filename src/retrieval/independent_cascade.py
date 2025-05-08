from .base import RAGAgent
from tools.embedding import EmbeddingPipeline
from tools.llm_output import ModelResponse
from ollama import Client
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Tuple, Set
import random
import json


class IndependentCascadeAgent(RAGAgent):
    """A RAG agent that uses Independent Cascade model for knowledge graph traversal.

    This agent combines embedding-based retrieval with probabilistic graph diffusion
    to identify relevant paths in a knowledge graph for question answering.

    Args:
        model_name (str, optional): Name of the LLM model. Defaults to "hermes3".
        neo4j_url (str): URL for Neo4j database connection.
        neo4j_username (str): Username for Neo4j authentication.
        neo4j_pw (str): Password for Neo4j authentication.
        embedding_model (str, optional): Name of embedding model. Defaults to "BAAI/bge-large-en-v1.5".
        number_of_iterations (int, optional): Maximum diffusion steps. Defaults to 5.
        number_of_simulations (int, optional): Number of Monte Carlo simulations. Defaults to 1000.
        activation_threshold (float, optional): Threshold for entity activation. Defaults to 0.40.
        max_edge_probability (float, optional): Maximum edge traversal probability. Defaults to 1.0.
    """

    def __init__(self,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 number_of_iterations: int = 10,
                 number_of_simulations: int = 1000,
                 activation_threshold: float = 0.40,
                 max_edge_probability: float = 1.0,
                 ollama_url: str = 'http://host.docker.internal:11434'):

        self.simulations = number_of_simulations
        self.iterations = number_of_iterations
        self.activation_threshold = activation_threshold
        self.max_edge_probability = max_edge_probability
        self.model_name = model_name
        self.client = Client(host=ollama_url, timeout=10*60)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = EmbeddingPipeline(embedding_model)

    def _retrieve_context(self, query_embedding: np.array, k_documents: int) -> List[Dict]:
        """Retrieves relevant context from the knowledge graph based on query embedding.

            Args:
                query_embedding (np.array): The embedding vector of the query.
                k_documents (int): Number of top documents to retrieve.

            Returns:
                List[Dict]: List of dictionaries containing head entity, relationship, tail entity information and value
                of cosine similarity between query and relationship embeddings.
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
                WITH e, other, r, gds.similarity.cosine(queryEmbedding, r.embedding) AS similarity
                ORDER BY similarity DESC
                WITH e, other, head(collect({name: r.name, similarity: similarity})) AS bestRel
                RETURN e.name AS head_entity_name, bestRel.name AS relationship_name, bestRel.similarity AS similarity, other.name AS tail_entity_name

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
                        WITH e, other, r, gds.similarity.cosine(queryEmbedding, r.embedding) AS similarity
                        ORDER BY similarity DESC
                        WITH e, other, head(collect({name: r.name, similarity: similarity})) AS bestRel
                        RETURN other.name AS head_entity_name, bestRel.name AS relationship_name, bestRel.similarity AS similarity, e.name AS tail_entity_name
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
        """Retrieves the most relevant entities based on embedding similarity.

            Args:
                query_embedding (np.array): The embedding vector of the query.
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
        """Retrieves detailed information about specified entities from the knowledge graph.

        Args:
            entity_list (List[str]): List of entity names to retrieve.

        Returns:
            List[Tuple[str, str]]: List of tuples containing:
                - Entity name
                - Concatenated entity attributes
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

    def _create_adj_dict(self, arc_list: List[Dict]) -> Dict[str, List[Tuple]]:
        """Creates bidirectional adjacency dictionary from relationship list.

        Args:
            arc_list (List[Dict]): List of relationship dictionaries.

        Returns:
            Dict[str, List[Tuple]]: Adjacency dictionary where each key maps to list of:
                - Connected entity name
                - Arc index
                - Edge probability
        """

        adj_dict = dict()
        for j, a in enumerate(arc_list):
            if not a['head_entity_name'] in adj_dict:
                adj_dict[a['head_entity_name']] = []
            if not a['tail_entity_name'] in adj_dict:
                adj_dict[a['tail_entity_name']] = []
            adj_dict[a['head_entity_name']].append(
                (a['tail_entity_name'], j, a['similarity'] * self.max_edge_probability))
            adj_dict[a['tail_entity_name']].append(
                (a['head_entity_name'], j, a['similarity'] * self.max_edge_probability))
        return adj_dict

    def _independent_cascade(self,
                             adj_dict: Dict[str, List[Tuple]],
                             initially_activated: List[str]
                             ):
        """Executes Independent Cascade diffusion model through Monte Carlo simulation.

        Args:
            adj_dict (Dict[str, List[Tuple]]): Graph adjacency dictionary.
            initially_activated (List[str]): Initially activated entity names.

        Returns:
            Tuple[Set[int], Set[str]]:
                - Set of relevant triplet indices
                - Set of activated entity names that exceeded threshold
        """
        # Initialize the counter for how many times each arc is used.
        entity_probability = {e: 0 for e in adj_dict.keys()}
        for e in initially_activated:
            entity_probability[e] = self.simulations
        for _ in range(self.simulations):
            # Use sets for fast membership checking.
            A = set(initially_activated)  # Nodes activated in the current time step.
            activated = set(initially_activated)  # All nodes activated so far.
            t = 0

            # Run the simulation for a fixed number of iterations.
            while t < self.iterations and A:
                newA = set()
                for u in A:
                    for arc in adj_dict[u]:
                        target, arc_index, prob = arc
                        if target in activated or target in newA:
                            continue
                        # For a proper probability check, you might use random.random().
                        if random.uniform(0, 1) < prob:
                            entity_probability[target] += 1
                            newA.add(target)
                A = newA
                activated |= newA
                t += 1

        for k in entity_probability:
            entity_probability[k] /= self.simulations

        activated_entities = {k for k, v in entity_probability.items() if v > self.activation_threshold}

        activated_triplets = {a[1] for e, arcs in adj_dict.items() if e in activated_entities
                              for a in arcs if a[0] in activated_entities and a[2] > 0.3}

        return activated_triplets, activated_entities

    def _generate_answer(self, query: str, knowledge: str) -> Dict:
        """Generates answer using LLM based on query and knowledge context.

        Args:
            query (str): Input question.
            knowledge (str): Retrieved knowledge context.

        Returns:
            str: Final answer from LLM without reasoning.
        """
        system_template = ("You are provided with the a multi-hop question."
                           "Additionally, you are given relationship statements that highlight"
                           "key connections between entities. Please answer the question based on the information from"
                           "relationship statements; Note that The answer to the question could be "
                           "either single word, yes/no or consist of multiple words describing single entity."
                           "Provide answer in JSON object with two string attributes, 'reasoning', which"
                           "provides your detailed reasoning about the answer, and"
                           "'final_answer' where you provide your short final answer without explaining your reasoning.")

        # Prepare messages for the model
        messages = [{"role": "system", "content": system_template}, {"role": "user", "content": knowledge},
                    {"role": "user", "content": query}]

        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                  format=ModelResponse.model_json_schema(), options={"temperature": 0.1})
        response = ModelResponse.validate(json.loads(result['message']['content']))
        return {'answer': response.final_answer, 'reasoning': response.reasoning, 'knowledge': knowledge}

    def generate_answer(self, query: str, retrieve_k: int = 10, key_entities: int = 10, ner_documents: int = 2) -> Dict:
        """Main method to generate answers from the knowledge graph.

             Args:
                query (str): The input question.
                retrieve_k (int, optional): Number of documents to retrieve. Defaults to 2.
                key_entities (int, optional): Number of initially activated entities. Defaults to 5.
                ner_documents (str): number of documents from which initially activated nodes are extracted

            Returns:
                str: Final answer generated from the knowledge graph.
        """
        # Generate query embedding and retrieve relevant documents
        query_embedding = self.embedding_pipeline.create_embedding(query)
        retrieved_rels = self._retrieve_context(query_embedding, retrieve_k)
        activated_entities = self._retrieve_activated_entities(query_embedding, ner_documents, key_entities)
        adj_dict = self._create_adj_dict(retrieved_rels)
        relevant_triplets, relevant_entities = self._independent_cascade(adj_dict=adj_dict,
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
