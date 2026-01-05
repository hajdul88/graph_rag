import pandas as pd
from typing import Optional, Union, Any, LiteralString
from .base_benchmark_adapter import BaseBenchmarkAdapter


class MultiHopRagAdapter(BaseBenchmarkAdapter):
    # TODO: make it download the data
    dataset_info = {
        "filename": "blablalalbal",
        "url_corpus": "/app/datasets/multihop-rag/corpus.json",
        "url_qa": "/app/datasets/multihop-rag/MultiHopRAG.json"
    }

    def load_corpus(
        self, limit: Optional[int] = None, seed: int = 42
    ) -> tuple[list[Union[LiteralString, str]], list[dict[str, Any]]]:
        filename = self.dataset_info["filename"]

        if filename is not None:
            corpus_df = pd.read_json(self.dataset_info['url_corpus'])
            qa_df = pd.read_json(self.dataset_info['url_qa'])
            qa_df['duplicate_count'] = qa_df.groupby(['answer']).cumcount()
            qa_df = qa_df[qa_df['duplicate_count'] < 4].reset_index().drop(columns='duplicate_count')
        else:
            print("Downloading data......")

        if limit is not None and 0 < limit < len(qa_df):
            qa_sample = qa_df.sample(n=limit, random_state=seed)
        else:
            qa_sample = qa_df

        corpus_list = []
        question_answer_pairs = []
        for _, row in qa_sample.iterrows():
            evidence_list = row['evidence_list']
            qa_pair = {
                'question': row['query'],
                'answer': row['answer']
            }
            for e in evidence_list:
                text = corpus_df[corpus_df['title'] == e['title']]['body'].item()
                corpus_list.append(text)
            question_answer_pairs.append(qa_pair)

        return corpus_list, question_answer_pairs
