from .naive_rag import BaselineAgent
from .independent_cascade import IndependentCascadeAgent
from .modified_diffusion_agent import DiffusionBFSAgent
from .linear_threshold import LinearThresholdAgent


class AgentFactory:
    @staticmethod
    def get_agent(neo4j_url: str, neo4j_username: str, neo4j_pw: str, agent_type: str, llm_model: str, ollama_url: str,
                  reasoning: bool, reasoning_prompt_loc: str, answering_prompt_loc: str):
        if agent_type == 'baseline':
            return BaselineAgent(neo4j_url=neo4j_url, neo4j_pw=neo4j_pw, neo4j_username=neo4j_username,
                                 model_name=llm_model, ollama_url=ollama_url)
        elif agent_type == 'independent_cascade':
            return IndependentCascadeAgent(neo4j_url=neo4j_url, neo4j_username=neo4j_username, neo4j_pw=neo4j_pw,
                                           model_name=llm_model, ollama_url=ollama_url)
        elif agent_type == 'modified_diffusion_agent':
            return DiffusionBFSAgent(neo4j_url=neo4j_url, neo4j_username=neo4j_username, neo4j_pw=neo4j_pw,
                                     model_name=llm_model, ollama_url=ollama_url,
                                     answering_prompt_loc=answering_prompt_loc,
                                     reasoning_prompt_loc=reasoning_prompt_loc,
                                     reasoning=reasoning)
        elif agent_type == 'linear_threshold':
            return LinearThresholdAgent(neo4j_url=neo4j_url, neo4j_username=neo4j_username, neo4j_pw=neo4j_pw,
                                        model_name=llm_model, ollama_url=ollama_url)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
