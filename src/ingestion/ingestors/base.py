from abc import ABC, abstractmethod
from typing import List


class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, corpus_list: List[str]):
        pass
