from typing import Generator


def size_based_chunking(text: str, chunk_size: int, overlap_size: int) -> Generator[str, None, None]:
    """Splits text into chunks of specified size with optional overlap between chunks.

        Args:
            text: Input text to be chunked.
            chunk_size: Size of each chunk in characters.
            overlap_size: Number of characters to overlap between consecutive chunks.

        Yields:
            str: Text chunks of specified size with overlap.

        Example:
            >>> text = "Hello world this is a test"
            >>> chunks = size_based_chunking(text, 10, 2)
            >>> next(chunks)
            'Hello worl'
        """

    chunk = text[:chunk_size]
    prev_chunk = ""  # For overlap storage
    i = 1
    while chunk:
        if prev_chunk:
            yield prev_chunk + chunk
        else:
            yield chunk

        # Store the last portion of the chunk for overlap with the next chunk
        prev_chunk = chunk[-overlap_size:] if overlap_size else ""

        # Read the next chunk and prepend the overlap from the previous chunk
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(text))
        chunk = text[start:end]
        i += 1


def word_based_chunking(text: str, chunk_size: int, overlap_size: int) -> Generator[str, None, None]:
    """Splits text into chunks based on word count with optional overlap between chunks.

        Args:
            text: Input text to be chunked.
            chunk_size: Number of words in each chunk.
            overlap_size: Number of words to overlap between consecutive chunks.

        Yields:
            str: Text chunks containing specified number of words with overlap.

        Raises:
            ValueError: If chunk_size is less than or equal to overlap_size.

        Example:
            >>> text = "Hello world this is a test"
            >>> chunks = word_based_chunking(text, 3, 1)
            >>> next(chunks)
            'Hello world this'
        """

    if chunk_size <= overlap_size:
        raise ValueError("chunk_size must be greater than overlap_size.")

    words = text.strip().split()
    start = 0
    while start <= len(words) - chunk_size:
        chunk = words[start:start + chunk_size]
        yield ' '.join(chunk)
        start += (chunk_size - overlap_size)
    # Handle the last chunk if there are remaining words
    if start < len(words):
        yield ' '.join(words[start:])
