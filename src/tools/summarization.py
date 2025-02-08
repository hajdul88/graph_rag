from huggingface_hub import InferenceClient
from ollama import Client


class SummaryPipeline:
    """
    A class for generating summaries of text using a language model.

    Attributes:
        model_name (str): The name of the language model to use for summarization.
        client (Client): An instance of the `Client` class from the `ollama` library for interacting with
            a remote language model server.
        template (dict): A template for the input to the language model.
    """
    def __init__(self, model_name: str = "hermes3"):
        self.model_name = model_name
        self.client = Client(host='http://host.docker.internal:11434')
        self.template = {"role": "system",
                         "content": "You summarize the text that user provides; Answer just with summary, "
                                    "no headlines no lists;"}

    def summarize(self, text: str, output_tokens: int = 300):
        """
            Generates a summary of the given text using the language model.

            Args:
                text (str): The text to summarize.
                output_tokens (int): The maximum number of tokens to use in the generated summary.

            Returns:
                str: The generated summary.
        """
        messages = [self.template, {"role": "user", "content": text}]
        result = self.client.chat(model=self.model_name, messages=messages, options={"max_tokens": output_tokens},
                                  stream=False)
        return result['message']['content']

