from typing import Dict
from pathlib import Path


class PromptManager:
    """Manages loading and caching of prompt templates."""

    def __init__(self):
        self._prompts: Dict[str, str] = {}

    def load_prompt(self, prompt_loc: str) -> str:
        """Load a prompt from file, with caching.
        
        Args:
            prompt_loc: Path to the prompt file.
            
        Returns:
            The prompt content as a string.
        """
        if prompt_loc not in self._prompts:
            with open(prompt_loc, 'r') as file:
                self._prompts[prompt_loc] = file.read()
        return self._prompts[prompt_loc]

    def get_prompts(self, answering_prompt_loc: str, reasoning_prompt_loc: str) -> tuple:
        """Load both answering and reasoning prompts.
        
        Args:
            answering_prompt_loc: Path to answering prompt. 
            reasoning_prompt_loc:  Path to reasoning prompt.
            
        Returns:
            Tuple of (answering_prompt, reasoning_prompt).
        """
        return (
            self.load_prompt(answering_prompt_loc),
            self.load_prompt(reasoning_prompt_loc)
        )
