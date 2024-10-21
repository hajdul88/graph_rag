from .text_reader import TextFileReader
from .pdf_reader import PDFFileReader

class FileReaderFactory:
    @staticmethod
    def get_reader(file_path: str):
        """Returns the appropriate file reader based on file type."""
        if file_path.endswith(".txt"):
            return TextFileReader()
        else:
            raise ValueError("Unsupported file type.")
