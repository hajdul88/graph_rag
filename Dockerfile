FROM python:3.9-slim

WORKDIR /app
# Install system dependencies including gcc
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

RUN poetry install --no-dev


COPY ./src /app/src
COPY ./tests /app/tests
COPY ./example_data /app/example_data
COPY ./files /app/files
COPY ./datasets /app/datasets
COPY ./results /app/results
COPY README.md /app/

WORKDIR /app/src

ENV txt_directory="../example_data/text"
ENV mode="size"
ENV chunk_size="300"
ENV overlap_size="20"
ENV txt_separator="\n\n"
ENV NEO4J_URI="bolt://neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="test1234"
ENV EMBED_MODEL="sentence-transformers/all-mpnet-base-v2"


CMD ["poetry", "run", "python", "run_pipeline.py"]
