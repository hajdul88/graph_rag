from abc import ABC, abstractmethod

class FileReader(ABC):
    @abstractmethod
    async def read(self, file_path: str) -> str:
        """Abstract method to asynchronously read a file."""
        pass