from typing import Optional

from evaluation_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter
from evaluation_framework.benchmark_adapters.musique_adapter import MusiqueQAAdapter


class CorpusBuilderExecutor:
    benchmark_adapter_options = {
        "TwoWikiMultiHop": TwoWikiMultihopAdapter,
        "MuSiQuE": MusiqueQAAdapter
    }

    benchmark_adapter = None
    raw_corpus = None
    questions = None

    def __init__(self, ingestion_pipeline, embedding_model="BAAI/bge-large-en-v1.5"):
        self.adapter = None
        self.ingestion_pipeline = ingestion_pipeline
        self.embedding_model = embedding_model

    async def build_corpus(self, limit: Optional[int] = None, benchmark="TwoWikiMultiHop", ingest: bool = False, ):
        if benchmark not in self.benchmark_adapter_options:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.adapter = self.benchmark_adapter_options[benchmark](embedding_model=self.embedding_model)
        self.raw_corpus, self.questions = self.adapter.load_corpus(limit=limit)
        if ingest:
            await self._ingest(self.raw_corpus)

        return self.questions, self.raw_corpus

    async def _ingest(self, doc_corpus):
        self.ingestion_pipeline.ingest(doc_corpus)
