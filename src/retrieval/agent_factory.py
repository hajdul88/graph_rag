from typing import Type, Dict
from .base import RAGAgent
from .retrieval_config import AgentConfig, Neo4jConfig, EmbeddingConfig, RetrievalConfig, ReasoningConfig
from .modified_diffusion_agent import DiffusionBFSAgent
from .baseline_cot import BaselineAgentCoT
from .hybrid_agent import HybridAgentCoT
from .decomposition_agent import DecompositionAgent

# Registry mapping agent types to their classes
AGENT_REGISTRY: Dict[str, Type[RAGAgent]] = {
    "baseline_cot": BaselineAgentCoT,
    "modified_diffusion_agent": DiffusionBFSAgent,
    "hybrid": HybridAgentCoT,
    "decomposition_agent": DecompositionAgent,
}


class AgentFactory:
    """Factory for creating RAG agents with configuration."""

    @staticmethod
    def get_agent(
            agent_type: str,
            model_name: str,
            neo4j_url: str,
            neo4j_username: str,
            neo4j_pw: str,
            llm_endpoint_url: str,
            answering_prompt_loc: str,
            reasoning_prompt_loc: str,
            reasoning: bool = True,
            reasoning_steps: int = 3,
            embedding_model_name: str = "BAAI/bge-large-en-v1.5",
            **agent_specific_kwargs
    ) -> RAGAgent:
        """Create a RAG agent with the specified configuration. 

        Args:
            agent_type: Type of agent to create (must be in AGENT_REGISTRY).
            model_name: Name of the LLM model. 
            neo4j_url:  Neo4j database URL.
            neo4j_username: Neo4j username.
            neo4j_pw: Neo4j password.
            llm_endpoint_url: LLM API URL.
            answering_prompt_loc: Path to answering prompt file.
            reasoning_prompt_loc: Path to reasoning prompt file. 
            reasoning: Enable reasoning steps.
            reasoning_steps: Maximum reasoning iterations.
            embedding_model_name: Name of the embedding model.
            **agent_specific_kwargs: Additional agent-specific parameters.

        Returns:
            An instance of the requested agent type. 

        Raises:
            ValueError:  If agent_type is not registered. 
        """
        if agent_type not in AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {list(AGENT_REGISTRY.keys())}"
            )

        config = AgentConfig(
            model_name=model_name,
            neo4j=Neo4jConfig(url=neo4j_url, username=neo4j_username, password=neo4j_pw),
            embedding=EmbeddingConfig(model_name=embedding_model_name),
            retrieval=RetrievalConfig(**{k: v for k, v in agent_specific_kwargs.items()
                                         if k in RetrievalConfig.__dataclass_fields__}),
            reasoning=ReasoningConfig(enabled=reasoning, max_steps=reasoning_steps),
            llm_endpoint_url=llm_endpoint_url,
            answering_prompt_loc=answering_prompt_loc,
            reasoning_prompt_loc=reasoning_prompt_loc,
        )

        agent_class = AGENT_REGISTRY[agent_type]
        return agent_class(config)
