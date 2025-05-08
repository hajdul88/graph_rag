from .basic_ingestor import BasicIngestor
from .knowledge_graph_ingestor import KnowledgeGraphIngestor
from .advanced_kg_ingestor import AdvancedKGIngestor


class IngestorFactory:

    @staticmethod
    def get_ingestor(ingestor_type: str, neo4j_url: str, neo4j_username: str, neo4j_pw: str,
                     ner_model: str = "qwen2.5:3b",
                     re_model: str = "hermes3",
                     re_template: str = "",
                     ner_template: str = "",
                     ollama_url: str = ""):
        if ingestor_type == 'basic':
            return BasicIngestor(neo4j_url=neo4j_url, neo4j_user=neo4j_username, neo4j_password=neo4j_pw)
        elif ingestor_type == 'knowledge_graph':
            return KnowledgeGraphIngestor(neo4j_url=neo4j_url, neo4j_user=neo4j_username, neo4j_password=neo4j_pw,
                                          template_ner_loc=ner_template, template_re_loc=re_template,
                                          ner_model=ner_model,
                                          re_model=re_model,
                                          ollama_url=ollama_url)
        elif ingestor_type == "advanced_knowledge_graph":
            return AdvancedKGIngestor(neo4j_url=neo4j_url, neo4j_user=neo4j_username, neo4j_password=neo4j_pw,
                                      template_ner_loc=ner_template, template_re_loc=re_template,
                                      ner_model=ner_model,
                                      re_model=re_model,
                                      ollama_url=ollama_url)
        else:
            raise ValueError(f"Unknown ingestor type: {ingestor_type}")
