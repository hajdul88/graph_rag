import numpy as np
import numpy.typing as npt
from typing import List, Dict
from itertools import combinations


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def create_adj_dict(relationship_triplets: List[List[str]]):
    adj_dict = dict()
    for t in relationship_triplets:
        if adj_dict.get(t[0]) is None:
            adj_dict[t[0]] = set()
        if adj_dict.get(t[2]) is None:
            adj_dict[t[2]] = set()
        adj_dict[t[0]].add(t[2])
        adj_dict[t[2]].add(t[0])
    return adj_dict


def create_embedding_dict(entity_list: List[Dict]):
    embedding_dict = dict()
    for e in entity_list:
        embedding_dict[e['entity_name']] = e['embedding']
    return embedding_dict


def find_similar_nodes(graph: Dict[str, set],
                       node_embeddings: Dict[str, npt.NDArray],
                       t_cosine: float,
                       t_jaccard: float):
    similar_nodes = []
    for node1, node2 in combinations(graph, 2):
        similarity_nb = jaccard_similarity(graph[node1], graph[node2])
        similarity_eb = cosine_similarity(node_embeddings[node1], node_embeddings[node2])
        if similarity_nb > t_jaccard and similarity_eb > t_cosine:
            similar_nodes.append((node1, node2))
    return similar_nodes
