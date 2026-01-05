from .base import RAGAgent
from tools.llm_output import ModelResponse
from neo4j import GraphDatabase
from tools.embedding import EmbeddingPipeline
from ollama import Client
from typing import List, Dict, Tuple
import torch
import numpy as np
import json
from collections import deque

reasoning_schema = {
    "type": "object",
    "properties": {
        "provided_context": {
            "type": "string",
        },
        "answer_possible": {
            "type": "boolean",
        },
        "final_answer": {
            "type": "string",
        },
        "additional_question": {
            "type": "string",
        }
    },
    "required": ["provided_context", "answer_possible", "final_answer", "additional_question"]
}


class HybridAgentCoT(RAGAgent):

    def __init__(self,
                 model_name: str = "hermes3",
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 answering_prompt_loc: str = "",
                 reasoning_prompt_loc: str = "",
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 ollama_url: str = 'http://host.docker.internal:11434',
                 max_reasoning_steps: int = 3,
                 k_hop: int = 4, ) -> None:

        self.model_name = model_name
        self.client = Client(host=ollama_url)
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = EmbeddingPipeline(embedding_model)
        with open(answering_prompt_loc, 'r') as file:
            self.answering_prompt = file.read()
        with open(reasoning_prompt_loc, 'r') as file:
            self.reasoning_prompt = file.read()
        self.max_reasoning_steps = max_reasoning_steps
        self.k_hop = k_hop

    def _retrieve_documents(self, query_embedding: torch.Tensor, k: int = 5) -> List[str]:
        """
        Retrieves the most relevant document chunks from Neo4j based on vector similarity.

        Uses vector search to find document chunks in the database that are most similar
        to the provided query embedding.

        Args:
            query_embedding: A tensor containing the embedding vector of the query.
            k: The number of top documents to retrieve. Defaults to 5.
        Returns:
            A list of text strings representing the retrieved document chunks.
        """
        with self.driver.session() as session:
            query_result = session.run("""
                            // Match chunks, calculate cosine similarity, and return top K results
                            WITH $query_embedding AS query_embedding 
                            CALL db.index.vector.queryNodes('textEmbedding', 500, query_embedding) 
                            YIELD node AS c, score AS similarity
                            RETURN c.text AS text, similarity
                            ORDER BY similarity DESC
                            LIMIT $top_k
                            """, query_embedding=query_embedding, top_k=k)
            return [d['text'] for d in query_result.data()]

    def _retrieve_seed_entities(self, query_embedding: np.array, top_k: int) -> List[str]:
        """Retrieves the most relevant entities based on the query embedding.

        Args:
            query_embedding: The embedding vector of the query.
            top_k: The number of top entities to retrieve.

        Returns:
            A list of entity names that are most relevant to the query.
        """

        query = """
                WITH $query_embedding AS queryEmbedding
                MATCH (d:Description)
                WITH d, gds.similarity.cosine(d.embedding, queryEmbedding) AS similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                MATCH (d)-[:DESCRIBES]->(e:Entity)
                RETURN DISTINCT e.name AS name
        """
        with self.driver.session() as session:
            results = session.run(query, query_embedding=query_embedding, top_k=top_k)
            return [r.data()['name'] for r in results]

    def _retrieve_k_hop_neighbours(self, seed_names: List[str], k_hop: int) -> List[str]:
        """Retrieve neighbors up to K hops away from seed entities.

            Args:
                seed_names: Initial list of entity names.
                k_hop: Maximum number of hops in the graph.

            Returns:
                Unique list of neighbor entity names within k_hop distance.
        """
        query = f"""
        UNWIND $seed_names AS seedName
        MATCH (s:Entity {{name: seedName}})
        OPTIONAL MATCH (s)-[:RELATED_TO*1..{k_hop}]->(neighbor:Entity)
        RETURN DISTINCT neighbor.name AS name
        """
        with self.driver.session() as session:
            results = session.run(query, seed_names=seed_names)
            return [r.data()['name'] for r in results]

    def _retrieve_entities(self, query_embedding: np.array, top_k: int) -> List[str]:
        """Combine seed and k-hop neighbor retrieval for entities.

            Args:
                query_embedding: Embedding vector of the query.
                top_k: Number of seed entities to retrieve.

            Returns:
                Combined list of unique entity names.
        """
        seed_names = self._retrieve_seed_entities(query_embedding, top_k)
        neighbor_names = self._retrieve_k_hop_neighbours(seed_names, self.k_hop)
        combined_entities = list(set(seed_names + neighbor_names))
        return combined_entities

    def _retrieve_relations(self, entity_names: List[str], query_embedding: np.array):
        """Retrieve relations among given entities, scored by similarity.

        Args:
            entity_names: Entities to consider as heads/tails.
            query_embedding: Embedding vector of the query.

        Returns:
            List of relation dicts with head, tail, name, and similarity score.
        """
        query = """
        WITH $entity_names AS e_names, $query_embedding AS queryEmbedding
        MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
        WHERE e1.name IN e_names AND e2.name IN e_names
        RETURN DISTINCT {
        head_entity_name: e1.name,
        tail_entity_name: e2.name,
        relationship_name: r.name,
        similarity: gds.similarity.cosine(queryEmbedding, r.embedding)
        } AS relation
        """
        with self.driver.session() as session:
            results = session.run(query, query_embedding=query_embedding, entity_names=entity_names)
            return [r.data()['relation'] for r in results]

    def _retrieve_activated_entities(self, query_embedding: np.array, activating_descriptions: int) -> List[str]:
        """Retrieves entities that are initially activated based on the query.

        Args:
            query_embedding: The embedding vector of the query.
            activating_descriptions: Number of top descriptions to consider for activation.

        Returns:
            A list of entity names that are initially activated.
        """
        query = """
                WITH $query_embedding AS queryEmbedding
                MATCH (d:Description)
                WITH d, gds.similarity.cosine(d.embedding, queryEmbedding) AS similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                MATCH (d)-[:DESCRIBES]->(e:Entity)
                RETURN DISTINCT e.name AS name
            """
        with self.driver.session() as s:
            results = s.run(query, query_embedding=query_embedding, top_k=activating_descriptions)
            return [record.data()['name'] for record in results]

    def _create_adj_dict(self, arc_list: List[Dict]) -> Dict[str, List[Tuple]]:
        """Creates an adjacency dictionary from the list of arcs.

        Args:
            arc_list: List of dictionaries containing relation information.

        Returns:
            A dictionary mapping entity names to lists of tuples containing
            (connected entity, arc index, similarity score).
        """

        adj_dict = dict()
        for j, a in enumerate(arc_list):
            if not a['head_entity_name'] in adj_dict:
                adj_dict[a['head_entity_name']] = []
            if not a['tail_entity_name'] in adj_dict:
                adj_dict[a['tail_entity_name']] = []
            adj_dict[a['head_entity_name']].append(
                (a['tail_entity_name'], j, a['similarity']))
            adj_dict[a['tail_entity_name']].append(
                (a['head_entity_name'], j, a['similarity']))
        return adj_dict

    def _diffusion_process(self,
                           adj_dict: Dict[str, List[Tuple]],
                           initially_activated: List[str],
                           activation_threshold,
                           pruning_threshold):
        """Performs the diffusion process to identify relevant entities and relationships.

        Args:
            adj_dict: Adjacency dictionary mapping entity names to their connections.
            initially_activated: List of initially activated entity names.
            activation_threshold: Threshold for entity activation.
            pruning_threshold: Threshold for pruning relationships.

        Returns:
            A tuple containing (set of relevant triplet indices, set of relevant entity names).
        """
        entity_score = {e: 0 for e in adj_dict.keys()}
        for e in initially_activated:
            entity_score[e] = max(1, entity_score[e])
            visited = set()
            queue = deque([e])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for arc in adj_dict[node]:
                    target, arc_index, prob = arc
                    entity_score[target] += prob * entity_score[node]
                    if target not in visited:
                        queue.append(target)
        activated_entities = {k for k, v in entity_score.items() if v > activation_threshold}
        relevant_triplets = {a[1] for e, arcs in adj_dict.items() if e in activated_entities
                             for a in arcs if a[0] in activated_entities and a[2] > pruning_threshold}
        return relevant_triplets, activated_entities

    def _retrieve_initial_context(self, query_embedding: np.array, retrieve_k: int, activating_descriptions: int,
                                  activation_threshold: float, pruning_threshold: float):
        """Run one step of retrieval and diffusion to assemble context.

            Args:
                query_embedding: Embedding vector of the query.
                retrieve_k: Number of seed entities to retrieve.
                activating_descriptions: Number of descriptions for initial activation.
                activation_threshold: Entity score threshold.
                pruning_threshold: Relation similarity threshold.

            Returns:
                A formatted context string with descriptions and relationships.
        """
        retrieved_entities = self._retrieve_entities(query_embedding, retrieve_k)
        retrieved_rels = self._retrieve_relations(retrieved_entities, query_embedding)
        activated_entities = self._retrieve_activated_entities(query_embedding, activating_descriptions)
        adj_dict = self._create_adj_dict(retrieved_rels)
        relevant_triplets, relevant_entities = self._diffusion_process(adj_dict=adj_dict,
                                                                       initially_activated=activated_entities,
                                                                       activation_threshold=activation_threshold,
                                                                       pruning_threshold=pruning_threshold)
        descriptions = self._retrieve_relevant_descriptions(list(relevant_entities), query_embedding)
        triplets_context = [' '.join((retrieved_rels[i]['head_entity_name'], retrieved_rels[i]['relationship_name'],
                                      retrieved_rels[i]['tail_entity_name']))
                            for i in relevant_triplets]
        context = ("### Context " + '\n' + '\n\n'.join(descriptions) +
                   '\n' + '**Relationships**' + '\n' + '\n'.join(triplets_context))
        return context

    def _generate_answer(self, query: str, context: str, retrieve_k: int) -> Dict:
        """
        """
        with open("", 'r') as file:
        for i in range(self.max_reasoning_steps):
            # Prepare messages for the model
            messages = [{"role": "system",
                         "content": self.reasoning_prompt if i >0 else },
                        {"role": "user", "content": query},
                        {"role": "user", "content": context}]
            # Generate and return response
            reasoning_result = self.client.chat(model=self.model_name, messages=messages, stream=False,
                                                keep_alive=0,
                                                format=reasoning_schema, options={"temperature": 0.1})
            reasoning_response = json.loads(reasoning_result['message']['content'])
            if reasoning_response['answer_possible']:
                break
            summarized_context = reasoning_response['provided_context']
            q = reasoning_response['additional_question']
            query_embedding = self.embedding_pipeline.create_embedding(q)
            documents = self._retrieve_documents(query_embedding, retrieve_k)
            context = '\n\n'.join(documents) + "\n\n" + summarized_context
        messages = [{"role": "system", "content": self.answering_prompt}, {"role": "user", "content": context},
                    {"role": "user", "content": query}]
        result = self.client.chat(model=self.model_name, messages=messages, stream=False, keep_alive=0,
                                  format=ModelResponse.model_json_schema(), options={"temperature": 0.1})
        response = ModelResponse.validate(json.loads(result['message']['content']))
        return {'answer': response.final_answer, 'reasoning': response.reasoning, 'knowledge': context}

    def generate_answer(self, query: str, retrieve_k: int = 10, activating_descriptions: int = 3,
                        activation_threshold: float = 0.5,
                        pruning_threshold: float = 0.3) -> Dict:
        """
        """
        # Generate query embedding and retrieve relevant documents
        query_embedding = self.embedding_pipeline.create_embedding(query)
        context = self._retrieve_initial_context(query_embedding, retrieve_k, activating_descriptions,
                                                 activation_threshold, pruning_threshold)
        retrieve_k = retrieve_k - 5
        # Generate and return answer
        return self._generate_answer(query, context, retrieve_k)
