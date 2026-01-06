import time
import logging
import asyncio
import json
from typing import Dict, Any

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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================================
# Utility Functions - File I/O
# ============================================================================

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with consistent error handling.

    Args:
        file_path: Path to the JSON file to load

    Returns:
        Parsed JSON content as dictionary

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If JSON is malformed
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {file_path}: {e}")


def save_json_file(file_path: str, data: Any) -> None:
    """
    Save data to JSON file.

    Args:
        file_path: Path where to save the JSON file
        data: Data to serialize as JSON
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============================================================================
# Utility Functions - Timing
# ============================================================================

def format_elapsed_time(start_time: float, end_time: float) -> str:
    """
    Convert elapsed time to HH:MM:SS format.

    Args:
        start_time: Start time from time.time()
        end_time: End time from time.time()

    Returns:
        Formatted time string in HH:MM:SS format
    """
    elapsed_seconds = end_time - start_time
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))


def log_execution_time(stage_name: str, start_time: float, end_time: float) -> None:
    """
    Log execution time for a pipeline stage.

    Args:
        stage_name: Name of the pipeline stage
        start_time: Start time from time.time()
        end_time: End time from time.time()
    """
    formatted_time = format_elapsed_time(start_time, end_time)
    logging.info(f"{stage_name} execution time: {formatted_time}")


# ============================================================================
# Component Initialization Functions
# ============================================================================

def create_database_orchestrator(config: DatabaseConfig) -> Orchestrator:
    """
    Initialize database orchestrator with Neo4j connection parameters.

    Args:
        config: Database configuration containing connection details

    Returns:
        Initialized Orchestrator instance
    """
    return Orchestrator(
        config.NEO4J_URL,
        config.NEO4J_USER,
        config.NEO4J_PASSWORD
    )


def create_ingestor(config_db: DatabaseConfig, config_llm: LLMConfig):
    """
    Initialize ingestor with database and LLM configurations.

    Args:
        config_db: Database configuration
        config_llm: LLM configuration

    Returns:
        Ingestor instance configured with provided parameters
    """
    return IngestorFactory.get_ingestor(
        ingestor_type=config_llm.ingestion_type,
        neo4j_url=config_db.NEO4J_URL,
        neo4j_username=config_db.NEO4J_USER,
        neo4j_pw=config_db.NEO4J_PASSWORD,
        ner_template=config_llm.NER_template,
        re_template=config_llm.RE_template,
        re_model=config_llm.model_name,
        ner_model=config_llm.model_name,
        ollama_url=config_llm.ollama_url,
        chunking_method=config_llm.chunking,
        chunk_size=config_llm.chunk_size_ingestion,
        overlap_size=config_llm.overlap_size_ingestion,
        embedding_model_name=config_llm.embedding_model_name
    )


def create_rag_agent(config_db: DatabaseConfig, config_llm: LLMConfig):
    """
    Initialize RAG agent with database and LLM configurations.

    Args:
        config_db: Database configuration
        config_llm:  LLM configuration

    Returns:
        RAG agent instance configured with provided parameters
    """
    return AgentFactory.get_agent(
        agent_type=config_llm.agent_type,
        neo4j_url=config_db.NEO4J_URL,
        neo4j_username=config_db.NEO4J_USER,
        neo4j_pw=config_db.NEO4J_PASSWORD,
        model_name=config_llm.model_name,
        ollama_url=config_llm.ollama_url,
        reasoning=config_llm.reasoning_enabled,
        reasoning_steps=config_llm.reasoning_steps,
        reasoning_prompt_loc=config_llm.reasoning_prompt_loc,
        answering_prompt_loc=config_llm.answering_prompt_loc,
        embedding_model_name=config_llm.embedding_model_name,
        k_hop=config_llm.K_HOP,
        retrieve_k=config_llm.RETRIEVE_K,
        activating_descriptions=config_llm.ACTIVATING_DESCRIPTIONS,
        activation_threshold=config_llm.ACTIVATION_THRESHOLD,
        pruning_threshold=config_llm.PRUNING_THRESHOLD,
        normalization_parameter=config_llm.NORMALIZATION_PARAMETER

    )


def create_graph_retriever(
        config_db: DatabaseConfig,
        config_file: FilePathConfig,
        config_llm: LLMConfig
) -> GraphRetriever:
    """
    Initialize graph retriever with standard configuration.

    Args:
        config_db: Database configuration
        config_file: File paths configuration
        config_llm: RAG parameters configuration

    Returns:
        Initialized GraphRetriever instance
    """
    return GraphRetriever(
        neo4j_url=config_db.NEO4J_URL,
        neo4j_username=config_db.NEO4J_USER,
        neo4j_pw=config_db.NEO4J_PASSWORD,
        graphs_folder=config_file.graphs_folder,
        normalization_parameter=config_llm.NORMALIZATION_PARAMETER
    )


# ============================================================================
# Pipeline Stage Functions
# ============================================================================

async def run_corpus_building_stage(
        config_eval: EvaluationConfig,
        config_file: FilePathConfig,
        config_llm: LLMConfig,
        ingestion_pipeline,
        db_orchestrator: Orchestrator
) -> None:
    """
    Execute corpus building pipeline stage.

    Builds a corpus from a benchmark dataset and optionally ingests it into
    the knowledge graph.  Saves questions and corpus to JSON files.

    Args:
        config_eval: Evaluation configuration controlling this stage
        config_file: File paths configuration for output files
        config_llm:  LLM configuration (unused, kept for consistency)
        ingestion_pipeline:  Initialized ingestor instance
        db_orchestrator: Database orchestrator for index creation
    """
    logging.info("Corpus Builder started...")
    start_time = time.time()

    # Build corpus from benchmark
    corpus_builder = CorpusBuilderExecutor(ingestion_pipeline=ingestion_pipeline,
                                           embedding_model=config_llm.embedding_model_name)
    questions, corpus = await corpus_builder.build_corpus(
        limit=config_eval.number_of_samples_in_corpus,
        benchmark=config_eval.benchmark,
        ingest=config_eval.ingest_corpus
    )

    # Save corpus and questions to files
    save_json_file(config_file.corpus_file, corpus)
    save_json_file(config_file.questions_file, questions)

    logging.info("Corpus Builder Ended...")
    end_time = time.time()
    log_execution_time("Ingestion", start_time, end_time)

    # Create database index
    db_orchestrator.create_index()
    logging.info("INGESTION FINISHED")


async def run_question_answering_stage(
        config_eval: EvaluationConfig,
        config_file: FilePathConfig,
        rag_agent
) -> None:
    """
    Execute question answering pipeline stage.

    Loads questions from file and generates answers using the RAG agent.
    Saves answers to JSON file.

    Args:
        config_eval: Evaluation configuration controlling this stage
        config_file: File paths configuration for input/output files
        rag_agent:  Initialized RAG agent instance
    """
    logging.info("Question answering started...")
    start_time = time.time()

    # Load questions from file
    questions = load_json_file(config_file.questions_file)
    questions = questions[: config_eval.number_of_samples_in_corpus]
    logging.info(f"Loaded {len(questions)} questions from {config_file.questions_file}")

    # Generate answers
    answer_generator = AnswerGeneratorExecutor(rag_agent)
    answers = await answer_generator.question_answering_non_parallel(questions=questions)

    # Save answers to file
    save_json_file(config_file.answers_file, answers)

    logging.info("Question answering ended...")
    end_time = time.time()
    log_execution_time("QA", start_time, end_time)


async def run_evaluation_stage(
        config_eval: EvaluationConfig,
        config_file: FilePathConfig
) -> None:
    """
    Execute evaluation pipeline stage.

    Loads answers from file and evaluates them using the configured
    evaluation engine and metrics.  Saves metrics to JSON file.

    Args:
        config_eval:  Evaluation configuration controlling this stage
        config_file: File paths configuration for input/output files
    """
    logging.info("Evaluation started...")

    # Load answers from file
    answers = load_json_file(config_file.answers_file)
    logging.info(f"Loaded {len(answers)} questions from {config_file.answers_file}")

    # Evaluate answers
    evaluator = EvaluationExecutor()
    metrics = await evaluator.execute(
        answers=answers,
        evaluator_engine=config_eval.evaluation_engine,
        evaluator_metrics=config_eval.evaluation_metrics,
    )

    # Save metrics to file
    save_json_file(config_file.metrics_file, metrics)

    logging.info("Evaluation ended...")


def run_dashboard_generation(
        config_eval: EvaluationConfig,
        config_file: FilePathConfig
) -> None:
    """
    Generate metrics dashboard if enabled.

    Args:
        config_eval: Evaluation configuration controlling this stage
        config_file: File paths configuration for input/output files
    """
    generate_metrics_dashboard(
        json_data=config_file.metrics_file,
        output_file=config_file.dashboard_file,
        benchmark=config_eval.benchmark
    )


async def run_graph_recording_stage(
        config_eval: EvaluationConfig,
        config_file: FilePathConfig,
        config_db: DatabaseConfig,
        config_llm: LLMConfig
) -> None:
    """
    Execute graph recording pipeline stage.

    Records knowledge subgraphs for a subset of questions, visualizing
    the retrieved context for each question.

    Args:
        config_eval:  Evaluation configuration controlling this stage
        config_file: File paths configuration
        config_db: Database configuration for graph retriever
        config_llm: RAG configuration for graph retriever
    """
    graph_retriever = create_graph_retriever(config_db, config_file, config_llm)

    # Load questions from file
    questions = load_json_file(config_file.questions_file)
    questions = questions[:config_eval.number_of_samples_in_corpus]

    # Record graphs for subset of questions
    for question in questions:
        question_id = question['id']

        # Skip questions not in visualization subset
        if question_id not in config_eval.questions_subset_vis:
            continue

        # Parse golden entities from question
        golden_entities = [entity.strip() for entity in question['entities'].split(" | ")]

        # Create and save graph for this question
        graph_retriever.create_and_save_graphs(
            query_id=question_id,
            query=question['question'],
            golden_entity_names=golden_entities,
            activation_threshold=config_llm.ACTIVATION_THRESHOLD,
            pruning_threshold=config_llm.PRUNING_THRESHOLD
        )


def run_cleanup_stage(
        config_eval: EvaluationConfig,
        db_orchestrator: Orchestrator
) -> None:
    """
    Clear database if cleanup is enabled.

    Args:
        config_eval: Evaluation configuration controlling this stage
        db_orchestrator: Database orchestrator for cleanup operations
    """
    db_orchestrator.clear_db()


# ============================================================================
# Main Orchestration
# ============================================================================

async def main():
    """
    Main orchestration function for the evaluation pipeline.

    Initializes all components and runs the enabled pipeline stages in order:
    1. Corpus building (if enabled)
    2. Question answering (if enabled)
    3. Evaluation (if enabled)
    4. Dashboard generation (if enabled)
    5. Graph recording (if enabled)
    6. Cleanup (if enabled)
    """
    # Load configurations
    config_eval = EvaluationConfig()
    config_file = FilePathConfig()
    config_db = DatabaseConfig()
    config_llm = LLMConfig()

    # Initialize components
    db_orchestrator = create_database_orchestrator(config_db)
    ingestion = create_ingestor(config_db, config_llm)
    rag_agent = create_rag_agent(config_db, config_llm)

    # Execute pipeline stages in order
    if config_eval.building_corpus_from_scratch:
        await run_corpus_building_stage(
            config_eval, config_file, config_llm, ingestion, db_orchestrator
        )

    if config_eval.answering_questions:
        await run_question_answering_stage(config_eval, config_file, rag_agent)

    if config_eval.evaluating_answers:
        await run_evaluation_stage(config_eval, config_file)

    if config_eval.dashboard:
        run_dashboard_generation(config_eval, config_file)

    if config_eval.record_context_graphs:
        await run_graph_recording_stage(config_eval, config_file, config_db, config_llm)

    if config_eval.delete_at_end:
        run_cleanup_stage(config_eval, db_orchestrator)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        logging.info("Done")
