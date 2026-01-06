import os
from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationConfig:
    """Controls which pipeline stages to run and how many samples."""
    building_corpus_from_scratch: bool = False
    ingest_corpus: bool = False
    number_of_samples_in_corpus: int = 1
    benchmark: str = "MuSiQuE"  # 'HotPotQA', 'TwoWikiMultiHop', 'MuSiQuE'
    answering_questions: bool = True
    evaluating_answers: bool = True
    evaluation_engine: str = "DeepEval"
    evaluation_metrics: List[str] = None
    dashboard: bool = True
    delete_at_end: bool = True
    record_context_graphs: bool = False
    # knowledge subgraph visualization option
    questions_subset_vis = ["2hop__67660_81007"]

    # subset of questions for knowledge subgraph visualization

    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["EM", "f1"]


@dataclass
class FilePathConfig:
    """File system paths for input/output."""
    questions_base: str = "/app/files/questions"
    answers_base: str = "/app/files/answers"
    results_base: str = "/app/results"
    templates_folder: str = "/app/files/templates"
    graphs_folder: str = "/app/files/graphs"

    questions_file_name: str = "musique_test"
    corpus_file_name: str = "musique_corpus_test"
    answers_file_name: str = "musique_test"

    @property
    def questions_file(self) -> str:
        return f"{self.questions_base}/{self.questions_file_name}_questions.json"

    @property
    def corpus_file(self) -> str:
        return f"{self.questions_base}/{self.corpus_file_name}_questions.json"

    @property
    def answers_file(self) -> str:
        return f"{self.answers_base}/{self.answers_file_name}_answers.json"

    @property
    def metrics_file(self) -> str:
        return f"{self.results_base}/{self.answers_file_name}_eval.json"

    @property
    def dashboard_file(self) -> str:
        return f"{self.results_base}/{self.answers_file_name}_dashboard.html"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    NEO4J_URL = os.environ['NEO4J_URL']  # 'NEO4J_URL' or 'NEO4J_NEW_URL' if you don't want to delete existing data
    NEO4J_USER = os.environ['NEO4J_USER']
    NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']


@dataclass
class LLMConfig:
    """Language model and RAG parameters."""
    ollama_url: str = os.environ['OLLAMA_URL']
    model_name: str = "phi4"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    templates_folder = '/app/files/templates'

    ingestion_type: str = "advanced_knowledge_graph"
    chunking: str = "word_based"
    chunk_size_ingestion: int = 500
    overlap_size_ingestion: int = 100
    NER_template: str = os.path.join(templates_folder, 'template_ner_v2.txt')
    RE_template = os.path.join(templates_folder, 'template_re_v2.txt')

    agent_type: str = "modified_diffusion_agent"
    reasoning_prompt_loc: str = None
    answering_prompt_loc: str = None
    reasoning_enabled: bool = True
    reasoning_steps: int = 3

    # Graph retrieval configuration thresholds
    ACTIVATION_THRESHOLD = 0.5
    PRUNING_THRESHOLD = 0.45
    NORMALIZATION_PARAMETER = 0.4
    K_HOP = 3
    RETRIEVE_K = 4
    ACTIVATING_DESCRIPTIONS = 4

    def __post_init__(self):
        if self.reasoning_prompt_loc is None:
            if self.agent_type == 'modified_diffusion_agent':
                self.reasoning_prompt_loc = os.path.join(self.templates_folder, 'reasoning_prompt.txt')
            elif self.agent_type in ['decomposition_agent', 'hybrid']:
                self.reasoning_prompt_loc = os.path.join(self.templates_folder, 'decompose_prompt.txt')
            else:
                self.reasoning_prompt_loc = os.path.join(self.templates_folder, 'reasoning_bl.txt')

        if self.answering_prompt_loc is None:
            if self.agent_type in ['modified_diffusion_agent', 'hybrid']:
                self.answering_prompt_loc = os.path.join(self.templates_folder, 'answering_prompt.txt')
            else:
                self.answering_prompt_loc = os.path.join(self.templates_folder, 'answering_bl.txt')
