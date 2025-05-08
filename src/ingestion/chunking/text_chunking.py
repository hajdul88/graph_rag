from typing import List, Generator, Tuple


def size_based_chunking(text: str, chunk_size: int, overlap_size: int) -> Generator[str, None, None]:
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


def chunk_documents(
        docs: List[str],
        chunk_method: str,
        chunk_size: int,
        overlap_size: int
) -> Generator[Tuple[int, str], None, None]:
    """
    Chunk each document in `docs` according to the specified method (`size_based` or `word_based`).
    Yields tuples of the form (doc_index, chunk_text).
    """
    for i, doc in enumerate(docs):
        if chunk_method == "size_based":
            for j, chunk in enumerate(size_based_chunking(doc, chunk_size, overlap_size)):
                yield i+j, chunk
        elif chunk_method == "word_based":
            for j, chunk in enumerate(word_based_chunking(doc, chunk_size, overlap_size)):
                yield i+j, chunk
        else:
            raise ValueError(f"Unknown chunking method: {chunk_method}")
