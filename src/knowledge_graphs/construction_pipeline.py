from ollama import Client
from typing import List, Any
import numpy as np
from .utils import *
import json
import time

struct_out_ner = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entity_name": {"type": "string"},
            "entity_type": {"type": "string"},
            "properties": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["entity_name", "entity_type", "properties"]
    }
}

struct_out_re = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "string"
                }
            }
        }
    },
    "required": ["triples"]
}


class KnowledgeGraphConstructor:
    """Constructs a knowledge graph by extracting entities and relationships from text.

       This class leverages language models and embedding pipelines to extract named entities and
       relationship triplets from a given text, compute embeddings for the relations, remove duplicates
       based on cosine similarity, and extract frequent relations based on occurrence counts.
    """

    def __init__(self, template_ner_loc: str, template_re_loc: str, embedding_pipeline: Any,
                 ner_model: str = "qwen2.5:7b", re_model: str = "qwen2.5:7b",
                 ollama_url: str = 'http://host.docker.internal:11434'):
        """Initializes the KnowledgeGraphConstructor.

                Loads the templates for named entity recognition (NER) and relationship extraction (RE)
                from the specified file locations. Also, sets up a client for interacting with a language
                model server and stores the provided embedding pipeline and NER model.

                Args:
                    template_ner_loc (str): The file path to the NER template.
                    template_re_loc (str): The file path to the RE template.
                    embedding_pipeline (EmbeddingPipeline): An instance of an embedding pipeline for creating
                        embeddings from textual data.
                    ner_model (str, optional): The language model identifier to be used for NER and RE.
                        Defaults to "hermes3".
        """
        self.client = Client(host=ollama_url, timeout=10*60)
        self.embedding_pipeline = embedding_pipeline
        with open(template_ner_loc, 'r') as file:
            self.template_ner = file.read()
        with open(template_re_loc, 'r') as file:
            self.template_re = file.read()
        self.ner_model = ner_model
        self.re_model = re_model

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extracts named entities from the given text using the NER template.

                This method sends a prompt to the language model client with the NER template and the
                provided text, and returns the model's response containing the extracted entities.

                Args:
                    text (str): The input text from which to extract entities.

                Returns:
                    str: A string representation of the extracted entities.
        """
        messages = [{"role": "system", "content": self.template_ner}, {"role": "user", "content": text}]
        try:
            result = self.client.chat(model=self.ner_model, messages=messages, stream=False, keep_alive=0,
                                      format=struct_out_ner)
        except Exception as e:
            print(e)
            time.sleep(20)
            result = self.client.chat(model=self.ner_model, messages=messages, stream=False, keep_alive=0,
                                      format=struct_out_ner)
        response = json.loads(result['message']['content'])
        return response

    def _create_entity_embeddings(self, entities: List[Dict]) -> List[Dict]:
        """Creates embeddings for a list of entities.

           Uses the embedding pipeline to generate vector representations for each entity
           and adds the embedding to the entity dictionary.

           Args:
               entities: A list of dictionaries containing entity information.

           Returns:
               List[Dict]: The input list of entity dictionaries, with an additional
                   'embedding' key containing the vector representation for each entity.
           """
        embeddings = [self.embedding_pipeline.create_embedding(str(e)) for e in entities]
        for i, em in enumerate(embeddings):
            entities[i]['embedding'] = em
        return entities

    def _create_relation_embeddings(self, triplets: List[List[str]]):
        """Creates embeddings for each relation triplet.

                Each triplet is converted to a string by joining its elements with a space, and then an
                embedding is computed using the provided embedding pipeline.

                Args:
                    triplets (List[List[str]]): A list of relation triplets, where each triplet is a list
                        of strings.

                Returns:
                    List[Any]: A list of embeddings corresponding to each triplet.
        """
        embeddings = [self.embedding_pipeline.create_embedding(' '.join(t)) for t in triplets]
        return embeddings

    def _extract_relationships(self, text: str, ner_list: str):
        """Extracts relationships from the given text by leveraging NER and RE templates.

                First, it extracts entities from the text using the NER template. Then, it appends the
                named entities to the text and sends the augmented text to the RE template for relationship
                extraction. The response is validated against a JSON schema, and only triplets with exactly
                three elements are considered. Finally, embeddings for these triplets are created.

                Args:
                    text (str): The input text from which to extract relationships.
                    ner_list (str): List of named entities

                Returns:
                    Tuple[List[List[str]], List[Any]]:
                        - A list of relationship triplets, where each triplet is a list of three strings.
                        - A list of embeddings corresponding to the triplets.
        """
        new_text = text + '\nnamed_entities: ' + ner_list
        messages = [{"role": "system", "content": self.template_re}, {"role": "user", "content": new_text}]
        try:
            result = self.client.chat(model=self.re_model, messages=messages, stream=False, keep_alive=0,
                                      format=struct_out_re)
        except Exception as e:
            print(e)
            time.sleep(20)
            result = self.client.chat(model=self.re_model, messages=messages, stream=False, keep_alive=0,
                                      format=struct_out_re)
        response = result['message']['content']
        kg = json.loads(response)
        relations = [t for t in kg['triples'] if len(t) == 3]
        embeddings = self._create_relation_embeddings(relations)
        return relations, embeddings

    def extract_frequent_relations(self, text: str, iterations: int, freq_tolerance: float = 0.5):
        """Extracts frequent relationship triplets from the given text over multiple iterations.

                Repeatedly extracts relationship triplets and their embeddings, counts the occurrences of
                entities in the triplets, and then filters out triplets that do not meet a frequency
                threshold based on the occurrences of their entities. Duplicate relations are removed
                using cosine similarity.

                Args:
                    text (str): The input text from which to extract frequent relationships.
                    iterations (int): The number of iterations to perform extraction, increasing the
                        robustness of frequency counts.
                    freq_tolerance (float, optional): The fraction of iterations an entity must appear in to
                        be considered frequent. Defaults to 0.5.

                Returns:
                    Tuple[List[List[str]], List[np.ndarray]]:
                        - A list of unique frequent relationship triplets.
                        - A list of corresponding embeddings for these unique triplets.
        """
        kg_entities = []
        kg_triplets = []
        kg_embeddings = []
        entity_count = dict()
        # save unique triplets and count extracted entities
        for i in range(0, iterations):
            entity_list = self._extract_entities(text)
            # append extracted entities to global list
            # extract relations
            names_list = [e['entity_name'] for e in entity_list]
            triplets, embeddings = self._extract_relationships(text, str(names_list))
            # append extracted embeddings and relations to global list
            kg_triplets.extend(triplets)
            kg_embeddings.extend(embeddings)
            # count entities
            for i, e in enumerate(names_list):
                if not (entity_count.get(e)):
                    entity_count[e] = 1
                    kg_entities.append(entity_list[i])
                else:
                    entity_count[e] += 1

        # extract KG triplets only for frequent entities
        filtered_triplets = []
        filtered_embeddings = []
        threshold = freq_tolerance * iterations
        freq_names = set(filter(lambda x: entity_count[x] > threshold, entity_count))
        freq_entities = [e for e in kg_entities if e['entity_name'] in freq_names]
        freq_entities = self._create_entity_embeddings(freq_entities)
        for i, r in enumerate(kg_triplets):
            if r[0] in freq_names and r[2] in freq_names:
                filtered_triplets.append(r)
                filtered_embeddings.append(kg_embeddings[i])
        return filtered_triplets, filtered_embeddings, freq_entities
