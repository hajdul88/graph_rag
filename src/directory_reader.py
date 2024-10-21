import os
import asyncio
from src.file_readers.factory import FileReaderFactory

class DirectoryFileReader:
    def __init__(self, directory_path: str, mode: str = "size", chunk_size: int = 100, txt_separator: str = None, overlap_size: int = 0):
        self.directory_path = directory_path
        self.mode = mode
        self.chunk_size = chunk_size
        self.txt_separator = txt_separator
        self.overlap_size = overlap_size

    async def read_files(self):
        """
        Asynchronously reads all .txt and .pdf files in the directory,
        yielding chunks based on the specified mode (size-based or separator-based),
        with optional overlap between chunks.

        Yields:
            str: The chunked content of each file.
        """
        if self.chunk_size <= self.overlap_size:
            raise ValueError(f"The chunk size {self.chunk_size} is smaller than or equal to the overlapping size {self.overlap_size}.")

        for file_name in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, file_name)

            try:
                reader = FileReaderFactory.get_reader(file_path)

                # Choose the separator based on the file type

                if file_name.endswith(".txt"):
                    separator = self.txt_separator
                else:
                    separator = None  # For unsupported file types, skip separator logic

                # Read the file with the appropriate separator
                async for chunk in reader.read(file_path, mode=self.mode, chunk_size=self.chunk_size, separator=separator, overlap_size=self.overlap_size):
                    yield chunk
            except ValueError:
                continue
                # Skip unsupported files or errors
