import json
from ollama import Client
from typing import List, Dict, Any
import time

struct_out_ner = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "aliases": {
                "type": "array",
                "items": {"type": "string"}
            },
            "entity_information": {"type": "string"}
        },
        "required": ["name", "type"]
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


class AdvancedKGConstructor:
    """Constructs a knowledge graph by extracting entities and relationships from text.

    This class leverages language models and embedding pipelines to extract named entities and
    relationship triplets from a given text, compute embeddings for the relations, remove duplicates
    based on cosine similarity, and extract frequent relations based on occurrence counts.
    """

    def __init__(self, template_ner_loc: str, template_re_loc: str, embedding_pipeline: Any,
                 ollama_url: str,
                 ner_model: str = "qwen2.5:3b", re_model: str = "hermes3"):
        """Initializes the KnowledgeGraphConstructor.

        Loads the templates for named entity recognition (NER) and relationship extraction (RE)
        from the specified file locations. Also, sets up a client for interacting with a language
        model server and stores the provided embedding pipeline and NER model.

        Args:
            template_ner_loc: The file path to the NER template.
            template_re_loc: The file path to the RE template.
            embedding_pipeline: An instance of an embedding pipeline for creating
                embeddings from textual data.
            ollama_url: URL for the Ollama API endpoint.
            ner_model: The language model identifier to be used for NER.
                Defaults to "qwen2.5:3b".
            re_model: The language model identifier to be used for RE.
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
            text: The input text from which to extract entities.

        Returns:
            A list of dictionaries representing the extracted entities with their properties.
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

        return json.loads(result['message']['content'])

    def _create_entity_embeddings(self, entities: List[Dict]) -> List[Dict]:
        """Creates embeddings for each entity based on its information.

        This method generates embeddings for each entity using the entity_information field
        and adds the embedding to the entity dictionary.

        Args:
            entities: A list of dictionaries representing entities.

        Returns:
            The list of entities with added embedding information.
        """
        embeddings = [self.embedding_pipeline.create_embedding(e['name'] + ": " + e['entity_information']) for e in entities]
        for i, em in enumerate(embeddings):
            entities[i]['embedding'] = em
        return entities

    def _create_relation_embeddings(self, triplets: List[List[str]]):
        """Creates embeddings for each relation triplet.

        Each triplet is converted to a string by joining its elements with a space, and then an
        embedding is computed using the provided embedding pipeline.

        Args:
            triplets: A list of relation triplets, where each triplet is a list
                of strings.

        Returns:
            A list of embeddings corresponding to each triplet.
        """
        embeddings = [self.embedding_pipeline.create_embedding(' '.join(t)) for t in triplets]
        return embeddings

    def _extract_relationships(self, text: str, ner_list: str):
        """Extracts relationships from the given text by leveraging NER and RE templates.

        First, it appends the named entities to the text and sends the augmented text to the RE
        template for relationship extraction. The response is validated against a JSON schema,
        and only triplets with exactly three elements are considered. Finally, embeddings for
        these triplets are created.

        Args:
            text: The input text from which to extract relationships.
            ner_list: List of named entities as a string.

        Returns:
            tuple: A tuple containing:
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

    def extract_entities_and_relationships(self, text: str):
        """Extracts entities and relationships from the given text.

        This method performs the full knowledge graph extraction pipeline:
        1. Extract named entities from the text
        2. Create embeddings for each entity
        3. Extract relationships between entities
        4. Create embeddings for each relationship

        Args:
            text: The input text from which to extract entities and relationships.

        Returns:
            tuple: A tuple containing:
                - A list of entities with their embeddings
                - A list of relationship triplets
                - A list of embeddings for the relationships
        """
        entities = self._extract_entities(text)
        entity_names = [e['name'] for e in entities]
        embedded_entities = self._create_entity_embeddings(entities)
        relationships, relation_embeddings = self._extract_relationships(text, ner_list=str(entity_names))
        return embedded_entities, relationships, relation_embeddings
