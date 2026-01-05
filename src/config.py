import os
from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationConfig:
    """Controls which pipeline stages to run and how many samples."""
    building_corpus_from_scratch: bool = False
    ingest_corpus: bool = False
    number_of_samples_in_corpus: int = 100
    benchmark: str = "TwoWikiMultiHop"  # 'HotPotQA', 'TwoWikiMultiHop', 'MuSiQuE'
    answering_questions: bool = True
    evaluating_answers: bool = True
    evaluation_engine: str = "DeepEval"
    evaluation_metrics: List[str] = None
    dashboard: bool = True
    delete_at_end: bool = False
    record_context_graphs: bool = False
    # knowledge subgraph visualization option
    questions_subset_vis = ["3hop2__668732_223623_162182", "4hop1__802394_153080_159767_81096",
                            "3hop1__488744_443779_52195", "4hop2__103790_14670_8987_8529",
                            "4hop1__199362_765799_282674_759393", "3hop1__198276_709625_84283",
                            "3hop1__67704_237521_291682", "4hop1__860115_798482_131926_13165",
                            "3hop1__307152_400692_51423",
                            "3hop1__274148_792411_51423", "3hop1__398232_326948_78782",
                            "4hop1__88342_75218_128008_67954",
                            "4hop3__312119_132409_371500_35031"]

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

    questions_file_name: str = "2wiki_100"
    corpus_file_name: str = "2wiki_corpus_100"
    answers_file_name: str = "test_phi"

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
        return f"{self.results_base}/{self.answers_file_name}_eval. json"

    @property
    def dashboard_file(self) -> str:
        return f"{self.results_base}/{self.answers_file_name}_dashboard.html"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    NEO4J_URL = os.environ['NEO4J_NEW_URL']
    NEO4J_USER = os.environ['NEO4J_USER']
    NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']


@dataclass
class LLMConfig:
    """Language model and RAG parameters."""
    ollama_url: str = os.environ['OLLAMA_URL']
    model_name: str = "phi4"
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
