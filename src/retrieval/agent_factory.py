from .modified_diffusion_agent import DiffusionBFSAgent
from .baseline_cot import BaselineAgentCoT
from .hybrid_agent import HybridAgentCoT
from .decomposition_agent import DecompositionAgent


class AgentFactory:
    @staticmethod
    def get_agent(neo4j_url: str, neo4j_username: str, neo4j_pw: str, agent_type: str, llm_model: str, ollama_url: str,
                  reasoning: bool, reasoning_steps: int, reasoning_prompt_loc: str, answering_prompt_loc: str):
        if agent_type == "baseline_cot":
            return BaselineAgentCoT(neo4j_url=neo4j_url, neo4j_pw=neo4j_pw, neo4j_username=neo4j_username,
                                    model_name=llm_model, ollama_url=ollama_url,
                                    answering_prompt_loc=answering_prompt_loc,
                                    reasoning_prompt_loc=reasoning_prompt_loc,
                                    reasoning=reasoning,
                                    max_reasoning_steps=reasoning_steps
                                    )
        elif agent_type == 'modified_diffusion_agent':
            return DiffusionBFSAgent(neo4j_url=neo4j_url, neo4j_username=neo4j_username, neo4j_pw=neo4j_pw,
                                     model_name=llm_model, ollama_url=ollama_url,
                                     answering_prompt_loc=answering_prompt_loc,
                                     reasoning_prompt_loc=reasoning_prompt_loc,
                                     reasoning=reasoning,
                                     max_reasoning_steps=reasoning_steps)
        elif agent_type == 'hybrid':
            return HybridAgentCoT(neo4j_url=neo4j_url, neo4j_pw=neo4j_pw, neo4j_username=neo4j_username,
                                  model_name=llm_model, ollama_url=ollama_url,
                                  answering_prompt_loc=answering_prompt_loc,
                                  reasoning_prompt_loc=reasoning_prompt_loc)
        elif agent_type == 'decomposition_agent':
            return DecompositionAgent(neo4j_url=neo4j_url, neo4j_pw=neo4j_pw, neo4j_username=neo4j_username,
                                      model_name=llm_model, ollama_url=ollama_url,
                                      answering_prompt_loc=answering_prompt_loc,
                                      reasoning_prompt_loc=reasoning_prompt_loc)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
