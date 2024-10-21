import aiofiles
from .base import FileReader


class TextFileReader(FileReader):
    async def read(self, file_path: str, mode: str = "size", chunk_size: int = 100, separator: str = None,
                   overlap_size: int = 0):
        """
        Asynchronously reads a text file and yields chunks based on the specified mode,
        with optional overlap between chunks. The overlap is included in both the previous
        and next chunks.

        Modes:
            - 'size': Yields chunks with a fixed number of characters, allowing overlap.
            - 'separator': Yields chunks split by the given separator, allowing overlap.

        Args:
            file_path (str): The path to the text file.
            mode (str): The chunking mode, either 'size' or 'separator'.
            chunk_size (int): Number of characters per chunk (used for size-based chunking).
            separator (str): A string used to split the content into chunks (used for separator-based chunking).
            overlap_size (int): The number of characters or lines that overlap between chunks.

        Yields:
            str: The chunked content based on the chosen mode.
        """
        if not file_path.endswith(".txt"):
            raise ValueError(f"{file_path} is not a text file.")

        if mode not in ["size", "separator"]:
            raise ValueError("Invalid mode. Choose 'size' or 'separator'.")

        async with aiofiles.open(file_path, mode='r') as file:
            if mode == "size":
                async for chunk in self._size_based_chunking(file, chunk_size, overlap_size):
                    yield chunk
            elif mode == "separator" and separator:
                async for chunk in self._separator_based_chunking(file, separator, overlap_size):
                    yield chunk

    async def _size_based_chunking(self, file, chunk_size: int, overlap_size: int):
        """
        Yields chunks of text based on the fixed number of characters, with optional overlap.
        The overlap is included in both the previous and next chunks.
        """
        chunk = await file.read(chunk_size)
        prev_chunk = ""  # For overlap storage

        while chunk:
            if prev_chunk:
                yield prev_chunk + chunk
            else:
                yield chunk

            # Store the last portion of the chunk for overlap with the next chunk
            prev_chunk = chunk[-overlap_size:] if overlap_size else ""

            # Read the next chunk and prepend the overlap from the previous chunk
            chunk = await file.read(chunk_size)

    async def _separator_based_chunking(self, file, separator: str, overlap_size: int):
        """
        Yields chunks of text split by a specified separator, with optional overlap.
        The overlap is included in both the previous and next chunks.
        """
        buffer = ""
        prev_chunk = ""  # For overlap storage

        async for line in file:
            buffer += line
            if separator in buffer:
                parts = buffer.split(separator)
                for part in parts[:-1]:
                    if prev_chunk:
                        yield prev_chunk + part
                    else:
                        yield part + separator

                    # Store the overlap
                    prev_chunk = part[-overlap_size:] + separator if overlap_size else ""

                buffer = parts[-1]

        # Yield any remaining content with overlap
        if buffer:
            if prev_chunk:
                yield prev_chunk + buffer
            else:
                yield buffer
