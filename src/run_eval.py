import time
import logging
import asyncio
import json
import os

from evaluation_framework.corpus_builder.corpus_builder_executor import CorpusBuilderExecutor
from evaluation_framework.answer_generation.answer_generation_executor import (
    AnswerGeneratorExecutor,
)
from evaluation_framework.evaluation.evaluation_executor import EvaluationExecutor
from tools.summarization import generate_metrics_dashboard
from tools.neo4j_tools import Orchestrator
from ingestion.ingestors.ingestor_factory import IngestorFactory
from retrieval.agent_factory import AgentFactory

eval_params = {
    # Corpus builder params
    "building_corpus_from_scratch": False,
    "number_of_samples_in_corpus": 100,
    "benchmark": "TwoWikiMultiHop",  # 'HotPotQA' or 'TwoWikiMultiHop' or 'MuSiQuE'
    # Question answering params
    "answering_questions": True,
    # Evaluation params
    "evaluating_answers": True,
    "evaluation_engine": "DeepEval",
    "evaluation_metrics": ["EM", "f1"],
    # Visualization
    "dashboard": True,
    # Clear database after experiment
    "delete_at_end": False
}

# File system parameters
QUESTIONS_FILE_NAME = "test_100"
ANSWERS_FILE_NAME = "test_phi"

questions_file = f"/app/files/questions/{QUESTIONS_FILE_NAME}_questions.json"
answers_file = f"/app/files/answers/{ANSWERS_FILE_NAME}_answers.json"
metrics_file = f"/app/results/{ANSWERS_FILE_NAME}_eval.json"
dashboard_file = f"/app/results/{ANSWERS_FILE_NAME}_dashboard.html"

# Database parameters
NEO4J_URL = os.environ['NEO4J_URL']
NEO4J_USER = os.environ['NEO4J_USER']
NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']

# RAG parameters
OLLAMA_URL = os.environ['OLLAMA_URL']
LLM_MODEL = 'phi4'
INGESTION = 'advanced_knowledge_graph'
AGENT = 'modified_diffusion_agent'
templates_folder = '/app/files/templates'
reasoning_flag = True


async def main():
    db_orchestrator = Orchestrator(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
    ingestion = IngestorFactory.get_ingestor(ingestor_type=INGESTION,
                                             neo4j_url=NEO4J_URL, neo4j_username=NEO4J_USER, neo4j_pw=NEO4J_PASSWORD,
                                             ner_template=os.path.join(templates_folder, 'template_ner_v2.txt'),
                                             re_template=os.path.join(templates_folder, 'template_re_v2.txt'),
                                             re_model="llama3.3",
                                             ner_model="llama3.3",
                                             ollama_url=OLLAMA_URL)

    rag_agent = AgentFactory.get_agent(agent_type=AGENT,
                                       neo4j_url=NEO4J_URL, neo4j_username=NEO4J_USER, neo4j_pw=NEO4J_PASSWORD,
                                       llm_model=LLM_MODEL,
                                       ollama_url=OLLAMA_URL,
                                       reasoning=reasoning_flag,
                                       reasoning_prompt_loc=os.path.join(templates_folder, 'reasoning_prompt.txt'),
                                       answering_prompt_loc=os.path.join(templates_folder, 'answering_prompt.txt'))
    ################################ Step 1: Corpus builder module
    if eval_params["building_corpus_from_scratch"]:
        logging.info("Corpus Builder started...")
        # Record start time
        start_time = time.time()
        # Build corpus
        corpus_builder = CorpusBuilderExecutor(ingestion_pipeline=ingestion)
        questions = await corpus_builder.build_corpus(
            limit=eval_params["number_of_samples_in_corpus"], benchmark=eval_params["benchmark"]
        )

        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        logging.info("Corpus Builder End...")
        # Record the end time and compute elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Log the elapsed time
        print("Ingestion execution time:", formatted_time)
        db_orchestrator.create_index()
    ################################ Step 2: Question answering module
    if eval_params["answering_questions"]:
        logging.info("Question answering started...")
        # Record start time
        start_time = time.time()
        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {questions_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {questions_file}: {e}")

        questions = questions[:eval_params["number_of_samples_in_corpus"]]
        print(f"Loaded {len(questions)} questions from {questions_file}")

        answer_generator = AnswerGeneratorExecutor(rag_agent)
        answers = await answer_generator.question_answering_non_parallel(questions=questions)

        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

        logging.info("Question answering end...")
        # Record the end time and compute elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Log the elapsed time
        print("QA execution time:", formatted_time)

    ################################ Step 3: Evaluation module
    if eval_params["evaluating_answers"]:
        logging.info("Evaluation started...")
        try:
            with open(answers_file, "r", encoding="utf-8") as f:
                answers = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {answers_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {answers_file}: {e}")

        print(f"Loaded {len(answers)} questions from {answers_file}")

        evaluator = EvaluationExecutor()
        metrics = await evaluator.execute(
            answers=answers,
            evaluator_engine=eval_params["evaluation_engine"],
            evaluator_metrics=eval_params["evaluation_metrics"],
        )

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        logging.info("Question answering end...")

    if eval_params["dashboard"]:
        generate_metrics_dashboard(
            json_data=metrics_file, output_file=dashboard_file,
            benchmark=eval_params["benchmark"]
        )

    if eval_params['delete_at_end']:
        db_orchestrator.clear_db()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        print("Done")
