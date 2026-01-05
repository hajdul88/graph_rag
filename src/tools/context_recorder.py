from tools.embedding import EmbeddingPipeline
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict
import networkx as nx
from collections import deque
import pickle


class GraphRetriever:
    def __init__(self,
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 k_hop: int = 4,
                 seed_entities: int = 3,
                 normalization_parameter: float = 0.4,
                 neo4j_url: str = "",
                 neo4j_username: str = "",
                 neo4j_pw: str = "",
                 graphs_folder: str = ""
                 ):
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
        self.embedding_pipeline = EmbeddingPipeline(embedding_model)
        self.k_hop = k_hop
        self.seed_k = seed_entities
        self.normalization_parameter = normalization_parameter
        self.graphs_folder = graphs_folder

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
        OPTIONAL MATCH (s)-[:RELATED_TO*1..{k_hop}]-(neighbor:Entity)
        RETURN DISTINCT neighbor.name AS name
        """
        with self.driver.session() as session:
            results = session.run(query, seed_names=seed_names)
            return [r.data()['name'] for r in results]

    def _retrieve_aliases(self, entity_names: List[str]):
        """
        Given canonical entity names, fetch all their aliases and return a combined list.
        Matching is case-insensitive and null-safe for `aliases`.

        Args:
            entity_names: List of entity *names* (not aliases).

        Returns List[str] of aliases.
        """
        if not entity_names:
            return []

        # Lowercase for case-insensitive matching
        lc_names = [n.lower() for n in entity_names if n is not None]

        query = """
        WITH $lc_names AS names
        MATCH (e:Entity)
        WHERE toLower(e.name) IN names
        WITH e, e.name AS name, coalesce(e.aliases, []) AS aliases
        RETURN name, aliases
        """
        with self.driver.session() as session:
            records = session.run(query, lc_names=lc_names)
            r_data = [r.data() for r in records]
            return {r['name']: r['aliases'] for r in r_data}

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

    def _create_graph(self, arc_list: List[Dict], initially_activated: List[str], golden: List[str],
                      normalization_parameter: float) -> nx.Graph:
        """Creates a NetworkX graph from the list of arcs.

        Args:
            arc_list: List of dictionaries containing relation information.
            initially_activated: List of initially activated entity names.

        Returns:
            A NetworkX Graph where nodes are entity names and edges include
            attributes: arc_index and similarity.
        """

        G = nx.Graph()

        # Add edges with attributes
        for j, a in enumerate(arc_list):
            head = a['head_entity_name']
            tail = a['tail_entity_name']
            sim = max(0.0, (a['similarity'] - normalization_parameter) / (1 - normalization_parameter))

            G.add_edge(head, tail, similarity=sim)

        # Ensure all initially activated nodes exist
        for n in initially_activated:
            if n not in G:
                G.add_node(n)
            G.nodes[n]['activated'] = True
        # Add or update activated nodes

        for n in G.nodes:
            if 'activated' not in G.nodes[n]:
                G.nodes[n]['activated'] = False
            if n in golden:
                G.nodes[n]['golden'] = True
            else:
                G.nodes[n]['golden'] = False
        return G

    def _construct_query_graph(self, query_id, query, golden_entity_names) -> nx.Graph:
        # gather data
        query_embedding = self.embedding_pipeline.create_embedding(query)
        seed_entities = self._retrieve_seed_entities(query_embedding, top_k=self.seed_k)
        neighbor_names = self._retrieve_k_hop_neighbours(seed_entities, self.k_hop)
        combined_entities = list(set(seed_entities + neighbor_names))
        retrieved_rels = self._retrieve_relations(combined_entities, query_embedding)

        aliases = self._retrieve_aliases(entity_names=combined_entities)
        golden_entities = []
        for name in golden_entity_names:
            for key, value in aliases.items():
                if name == key or name in value:
                    golden_entities.append(key)
        G = self._create_graph(retrieved_rels, seed_entities, golden_entities, self.normalization_parameter)
        G.graph["description"] = query
        with open(f"{self.graphs_folder}/{query_id}_retrieved.pkl", "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        return G

    def _diffusion_process(
            self,
            G: nx.Graph,
            query_id: str,
            query: str,
            activation_threshold: float,
            pruning_threshold: float,
    ) -> None:
        """
        Performs the diffusion process directly on a NetworkX graph.
        """

        query_embedding = self.embedding_pipeline.create_embedding(query)
        initially_activated = self._retrieve_seed_entities(query_embedding, top_k=self.seed_k)

        # Initialize scores for all nodes
        entity_score = {n: 0.0 for n in G.nodes}

        # Diffusion from each seed using edge weight 'similarity'
        for seed in initially_activated:
            entity_score[seed] = max(1.0, entity_score[seed])
            visited = set()
            queue = deque([seed])

            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)

                # Traverse neighbors; edge data holds 'similarity'
                for nbr in G.neighbors(node):
                    edge_data = G[node][nbr]
                    prob = edge_data.get('similarity', 0.0)
                    entity_score[nbr] += min(1.0, prob * entity_score[node])
                    if nbr not in visited:
                        queue.append(nbr)

        # Mark nodes as activated based on threshold
        for n in G.nodes:
            G.nodes[n]['activated'] = entity_score[n] > activation_threshold

        # Mark edges as relevant if both endpoints activated and similarity passes pruning threshold
        for u, v, data in G.edges(data=True):
            sim = data.get('similarity', 0.0)
            data['relevant'] = (
                    G.nodes[u].get('activated', False)
                    and G.nodes[v].get('activated', False)
                    and sim >= pruning_threshold
            )

        with open(f"{self.graphs_folder}/{query_id}_diffusion_{self.normalization_parameter}_{activation_threshold}.pkl", "wb") as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        # Function mutates G in place; no return
        return

    def create_and_save_graphs(self, query_id, query, golden_entity_names, activation_threshold,  pruning_threshold):
        G = self._construct_query_graph(query_id, query, golden_entity_names)
        self._diffusion_process(G, query_id, query, activation_threshold,  pruning_threshold)

