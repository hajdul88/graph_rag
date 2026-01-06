from .basic_ingestor import BasicIngestor
from .advanced_kg_ingestor import AdvancedKGIngestor


class IngestorFactory:

    @staticmethod
    def get_ingestor(ingestor_type: str, neo4j_url: str, neo4j_username: str, neo4j_pw: str,
                     ner_model: str = "qwen2.5:3b",
                     re_model: str = "hermes3",
                     re_template: str = "",
                     ner_template: str = "",
                     ollama_url: str = "",
                     chunking_method: str = "word_based",
                     chunk_size: int = 500,
                     overlap_size: int = 100,
                     embedding_model_name: str = "BAAI/bge-large-en-v1.5"):
        if ingestor_type == 'basic':
            return BasicIngestor(neo4j_url=neo4j_url, neo4j_user=neo4j_username, neo4j_password=neo4j_pw,
                                 chunking_method=chunking_method, chunk_size=chunk_size, overlap_size=overlap_size,
                                 model_name_embedding=embedding_model_name)
        elif ingestor_type == "advanced_knowledge_graph":
            return AdvancedKGIngestor(neo4j_url=neo4j_url, neo4j_user=neo4j_username, neo4j_password=neo4j_pw,
                                      template_ner_loc=ner_template, template_re_loc=re_template,
                                      ner_model=ner_model,
                                      re_model=re_model,
                                      ollama_url=ollama_url,
                                      chunking_method=chunking_method,
                                      chunk_size=chunk_size,
                                      overlap_size=overlap_size,
                                      model_name_embedding=embedding_model_name)
        else:
            raise ValueError(f"Unknown ingestor type: {ingestor_type}")
