from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from typing import List
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
        return self.pipeline(text, truncation=True)[0][0]


class RerankingPipeline:
    """
        A pipeline for reranking documents based on relevance to a query using a pretrained model.

        Attributes:
            tokenizer (AutoTokenizer): The tokenizer used to preprocess text inputs.
            model (AutoModelForSequenceClassification): The pretrained model for reranking.

        Args:
            model_name (str): The name of the pretrained model to load. Defaults to "BAAI/bge-reranker-large".

        Methods:
            rerank(query: str, documents: List[str]) -> List[int]:
                Reranks the provided list of documents based on relevance to the query.

        Example:
            pipeline = RerankingPipeline()
            query = "What is the capital of France?"
            documents = ["Paris is the capital of France.",
                        "Berlin is the capital of Germany.",
                        "Madrid is the capital of Spain."]
            ranks = pipeline.rerank(query, documents)
            print(ranks)  # Example output: [0, 2, 1]
        """
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query: str, documents: List[str]) -> List[int]:
        """
                Ranks the provided list of documents based on their relevance to the query.

                Args:
                    query (str): The query to evaluate relevance against.
                    documents (List[str]): A list of documents to rerank.

                Returns:
                    List[int]: A list of indices representing the documents ranked by relevance,
                               sorted in descending order of relevance.

                Raises:
                    ValueError: If the documents list is empty.
                """
        if not documents:
            raise ValueError("The documents list cannot be empty.")

        pairs = [(query, txt) for txt in documents]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return torch.sort(scores, descending=True).indices.tolist()
