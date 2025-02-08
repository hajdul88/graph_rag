from .document_reader import word_based_chunking, size_based_chunking
from pandas import DataFrame
from tools.summarization import SummaryPipeline


class DatasetReader:
    """A class for processing and chunking text data from a DataFrame for RAG applications.

    This class handles text data preprocessing by reading from a DataFrame and splitting content
    into manageable chunks using either word-based or size-based chunking strategies. It supports
    optional text summarization and flexible chunk overlapping for better context preservation.

    Attributes:
        df (DataFrame): Source DataFrame containing 'title' and 'text' columns.
        mode (str): Chunking strategy - 'word' for word-based or 'size' for character-based chunking.
        chunk_size (int): Number of words or characters per chunk depending on mode.
        txt_separator (str, optional): Custom separator for text concatenation.
        overlap_size (int): Overlap size between consecutive chunks for context continuity.
        summarize (bool): Whether to generate text summaries.
        summary_pipeline (SummaryPipeline, optional): Pipeline for text summarization if enabled.
    """

    def __init__(self, dataframe: DataFrame, mode: str = "word", chunk_size: int = 500, overlap_size: int = 100,
                 txt_separator: str = None, summarize: bool = False):
        """Initialize the DatasetReader.

                Args:
                    dataframe: Input DataFrame containing 'title' and 'text' columns.
                    mode: Chunking strategy, either 'word' or 'size'.
                    chunk_size: Size of each chunk (words or characters).
                    overlap_size: Size of overlap between chunks.
                    txt_separator: Optional separator for text concatenation.
                """
        self.df = dataframe
        self.mode = mode
        self.chunk_size = chunk_size
        self.txt_separator = txt_separator
        self.overlap_size = overlap_size
        self.summarize = summarize
        if summarize:
            self.summary_pipeline = SummaryPipeline()

    def read_files(self):
        """Read and chunk text data from the DataFrame.

        Yields:
            tuple: A pair of (title, chunk) where title is the document title
                  and chunk is a text segment.

        Raises:
            ValueError: If chunk_size is less than or equal to overlap_size.
        """
        if self.chunk_size <= self.overlap_size:
            raise ValueError(
                f"The chunk size {self.chunk_size} is smaller than or equal to the overlapping size {self.overlap_size}.")

        for _, data in self.df.iterrows():
            title = data['title']
            if self.summarize:
                summary = self.summary_pipeline.summarize(data['text'])
            if self.mode == "word":
                for chunk in word_based_chunking(data['text'], self.chunk_size, self.overlap_size):
                    if self.summarize:
                        yield title, chunk, summary
                    else:
                        yield title, chunk
            elif self.mode == "size":
                for chunk in size_based_chunking(data['text'], self.chunk_size, self.overlap_size):
                    if self.summarize:
                        yield title, chunk, summary
                    else:
                        yield title, chunk
