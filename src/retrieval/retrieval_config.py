from dataclasses import dataclass, field


@dataclass
class Neo4jConfig:
    """Neo4j connection configuration."""
    url: str
    username: str
    password: str


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "BAAI/bge-large-en-v1.5"


@dataclass
class RetrievalConfig:
    """Graph retrieval parameters."""
    k_hop: int = 3
    retrieve_k: int = 10
    activating_descriptions: int = 10
    activation_threshold: float = 0.5
    pruning_threshold: float = 0.5


@dataclass
class ReasoningConfig:
    """LLM reasoning parameters."""
    enabled: bool = True
    max_steps: int = 3
    temperature: float = 0.0


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    model_name: str
    neo4j: Neo4jConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    ollama_url: str = 'http://host.docker.internal:11434'
    answering_prompt_loc: str = ""
    reasoning_prompt_loc: str = ""
