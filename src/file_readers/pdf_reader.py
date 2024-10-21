import fitz
from .base import FileReader


class PDFFileReader(FileReader):
    async def read(self, file_path: str, mode: str = "size", chunk_size: int = 100, separator: str = None,
                   overlap_size: int = 0):
        """
        Placeholder method for reading a PDF file. Raises NotImplementedError for now,
        indicating that the method is not yet implemented.

        Args:
            file_path (str): The path to the PDF file.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("The 'read' method for PDFFileReader is not yet implemented.")

