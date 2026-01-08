FROM python:3.11-slim

WORKDIR /app
# Install system dependencies including gcc
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry==1.6.1

RUN poetry install --no-dev


COPY ./src /app/src
COPY ./tests /app/tests
COPY ./files /app/files
COPY ./datasets /app/datasets
COPY ./results /app/results
COPY README.md /app/

WORKDIR /app/src

ENV NEO4J_URL="bolt://neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="test1234"
ENV LLM_ENDPOINT_URL = ""

CMD ["poetry", "run", "python", "run_eval.py"]
