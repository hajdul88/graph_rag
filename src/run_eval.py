import time
import logging
import asyncio
import json

from config import EvaluationConfig, FilePathConfig, DatabaseConfig, LLMConfig

from evaluation_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from evaluation_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)
from evaluation_framework.evaluation.evaluation_executor import EvaluationExecutor
from tools.summarization import generate_metrics_dashboard
from tools.neo4j_tools import Orchestrator
from ingestion.ingestors.ingestor_factory import IngestorFactory
from retrieval.agent_factory import AgentFactory
from tools.context_recorder import GraphRetriever




async def main():
    evaluation_config = EvaluationConfig()
    file_path_config = FilePathConfig()
    database_config = DatabaseConfig()
    llm_config = LLMConfig()
    db_orchestrator = Orchestrator(database_config.NEO4J_URL, database_config.NEO4J_USER, database_config.NEO4J_PASSWORD)
    ingestion = IngestorFactory.get_ingestor(ingestor_type=llm_config.ingestion_type,
                                             neo4j_url=database_config.NEO4J_URL,
                                             neo4j_username=database_config.NEO4J_USER,
                                             neo4j_pw=database_config.NEO4J_PASSWORD,
                                             ner_template=llm_config.NER_template,
                                             re_template=llm_config.RE_template,
                                             re_model=llm_config.model_name,
                                             ner_model=llm_config.model_name,
                                             ollama_url=llm_config.ollama_url,
                                             chunking_method=llm_config.chunking,
                                             chunk_size=llm_config.chunk_size_ingestion,
                                             overlap_size=llm_config.overlap_size_ingestion)

    rag_agent = AgentFactory.get_agent(agent_type=llm_config.agent_type,
                                       neo4j_url=database_config.NEO4J_URL, neo4j_username=database_config.NEO4J_USER,
                                       neo4j_pw=database_config.NEO4J_PASSWORD,
                                       llm_model=llm_config.model_name,
                                       ollama_url=llm_config.ollama_url,
                                       reasoning=llm_config.reasoning_enabled,
                                       reasoning_steps=llm_config.reasoning_steps,
                                       reasoning_prompt_loc=llm_config.reasoning_prompt_loc,
                                       answering_prompt_loc=llm_config.answering_prompt_loc)

    ################################ Step 1: Corpus builder module
    if evaluation_config.building_corpus_from_scratch:
        logging.info("Corpus Builder started...")
        # Record start time
        start_time = time.time()
        # Build corpus
        corpus_builder = CorpusBuilderExecutor(ingestion_pipeline=ingestion)
        questions, corpus = await corpus_builder.build_corpus(
            limit=evaluation_config.number_of_samples_in_corpus, benchmark=evaluation_config.benchmark,
            ingest=evaluation_config.ingest_corpus
        )
        with open(file_path_config.corpus_file, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        with open(file_path_config.questions_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        logging.info("Corpus Builder End...")
        # Record the end time and compute elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Log the elapsed time
        print("Ingestion execution time:", formatted_time)
        db_orchestrator.create_index()
        print("INGESTION FINISHED")
    ################################ Step 2: Question answering module
    if evaluation_config.answering_questions:
        logging.info("Question answering started...")
        # Record start time
        start_time = time.time()
        try:
            with open(file_path_config.questions_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {file_path_config.questions_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path_config.questions_file}: {e}")

        questions = questions[:evaluation_config.number_of_samples_in_corpus]
        print(f"Loaded {len(questions)} questions from {file_path_config.questions_file}")

        answer_generator = AnswerGeneratorExecutor(rag_agent)
        answers = await answer_generator.question_answering_non_parallel(questions=questions)

        with open(file_path_config.answers_file, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

        logging.info("Question answering end...")
        # Record the end time and compute elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Log the elapsed time
        print("QA execution time:", formatted_time)

    ################################ Step 3: Evaluation module
    if evaluation_config.evaluating_answers:
        logging.info("Evaluation started...")
        try:
            with open(file_path_config.answers_file, "r", encoding="utf-8") as f:
                answers = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {file_path_config.answers_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path_config.answers_file}: {e}")

        print(f"Loaded {len(answers)} questions from {file_path_config.answers_file}")

        evaluator = EvaluationExecutor()
        metrics = await evaluator.execute(
            answers=answers,
            evaluator_engine=evaluation_config.evaluation_engine,
            evaluator_metrics=evaluation_config.evaluation_metrics,
        )

        with open(file_path_config.metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        logging.info("Question answering end...")

    if evaluation_config.dashboard:
        generate_metrics_dashboard(
            json_data=file_path_config.metrics_file, output_file=file_path_config.dashboard_file,
            benchmark=evaluation_config.benchmark
        )

    ################################### RECORD CONTEXT GRAPHS
    if evaluation_config.record_context_graphs:
        graph_retriever = GraphRetriever(
            neo4j_url=database_config.NEO4J_URL,
            neo4j_username=database_config.NEO4J_USER,
            neo4j_pw=database_config.NEO4J_PASSWORD,
            graphs_folder=file_path_config.graphs_folder,
            normalization_parameter=0.4
        )
        try:
            with open(file_path_config.questions_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {file_path_config.questions_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path_config.questions_file}: {e}")

        questions = questions[:evaluation_config.number_of_samples_in_corpus]
        for q in questions:
            if q['id'] not in evaluation_config.questions_subset_vis:
                continue
            golden_entities = [e.strip() for e in q['entities'].split(" | ")]
            graph_retriever.create_and_save_graphs(query_id=q['id'],
                                                   query=q['question'],
                                                   golden_entity_names=golden_entities,
                                                   activation_threshold=0.5,
                                                   pruning_threshold=0.45)

    ################################### DELETE
    if evaluation_config.delete_at_end:
        db_orchestrator.clear_db()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
