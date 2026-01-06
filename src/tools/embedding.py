from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from typing import List
import numpy as np
import torch


class EmbeddingPipeline:
    """Creates embeddings for text using pre-trained transformer models.

    This class provides functionality to generate vector embeddings from text input
    using transformer models from the Hugging Face ecosystem.

    Attributes:
        tokenizer: An instance of AutoTokenizer for text tokenization.
        pipeline: A transformers Pipeline object for feature extraction.

    Example:
        embedding_pipeline = EmbeddingPipeline()
        text = "Hello, world!"
        embedding = embedding_pipeline.create_embedding(text)
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """Initializes the EmbeddingPipeline with a specified model.

        Args:
            model_name (str): The name of the pre-trained model to use.
            Defaults to "BAAI/bge-large-en-v1.5".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "feature-extraction",
            model=model_name,
            tokenizer=self.tokenizer,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def create_embedding(self, text: str):
        """Creates an embedding vector for the input text.

        Args:
            text (str): The input text to create embeddings for.

        Returns:
            numpy.ndarray: A vector representation of the input text.
        """
        return np.array(self.pipeline(text, truncation=True)[0][0])