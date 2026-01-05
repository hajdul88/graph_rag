from typing import Optional

from evaluation_framework.benchmark_adapters.hotpot_qa_adapter import HotpotQAAdapter
from evaluation_framework.benchmark_adapters.twowikimultihop_adapter import TwoWikiMultihopAdapter
from evaluation_framework.benchmark_adapters.musique_adapter import MusiqueQAAdapter
from evaluation_framework.benchmark_adapters.multihop_rag_adapter import MultiHopRagAdapter

class CorpusBuilderExecutor:
    benchmark_adapter_options = {
        "HotPotQA": HotpotQAAdapter,
        "TwoWikiMultiHop": TwoWikiMultihopAdapter,
        "MuSiQuE": MusiqueQAAdapter,
        "MultiHopRAG": MultiHopRagAdapter
    }

    benchmark_adapter = None
    raw_corpus = None
    questions = None

    def __init__(self, ingestion_pipeline):
        self.adapter = None
        self.ingestion_pipeline = ingestion_pipeline

    async def build_corpus(self, limit: Optional[int] = None, benchmark="HotPorQA", ingest: bool = False,):
        print(benchmark)
        if benchmark not in self.benchmark_adapter_options:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        self.adapter = self.benchmark_adapter_options[benchmark]()
        self.raw_corpus, self.questions = self.adapter.load_corpus(limit=limit)
        if ingest:
            await self._ingest(self.raw_corpus)

        return self.questions, self.raw_corpus

    async def _ingest(self, doc_corpus):
        self.ingestion_pipeline.ingest(doc_corpus)
