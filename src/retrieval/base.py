from abc import ABC, abstractmethod
from typing import Dict
from .retrieval_config import AgentConfig


class RAGAgent(ABC):
    """
    Abstract base class for RAG retrieval performing document retrieval and answer generation. 

    Attributes:
        config: AgentConfig instance with all agent parameters.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the RAG agent with configuration. 

        Args:
            config: AgentConfig instance. 
        """
        self.config = config

    @abstractmethod
    def generate_answer(self, query: str, **kwargs) -> Dict:
        """
        Main method for generating answers: 
        1. Create query embedding. 
        2. Retrieve relevant documents.
        3. Generate and return an answer based on retrieved content.

        Args:
            query: The user's question. 
            **kwargs: Additional agent-specific parameters.

        Returns:
            Dictionary containing 'answer', 'reasoning', and 'knowledge' keys.
        """
        pass
