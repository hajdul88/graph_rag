FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

RUN poetry install --no-dev


COPY ./src /app/src
COPY ./tests /app/tests
COPY ./example_data /app/example_data
COPY README.md /app/

WORKDIR /app/src

ENV directory="../example_data"
ENV mode="size"
ENV chunk_size="300"
ENV overlap_size="20"
ENV txt_separator="\n\n"
ENV NEO4J_URI="bolt://neo4j:7687"
ENV NEO4J_USER="neo4j"
ENV NEO4J_PASSWORD="test1234"


CMD ["poetry", "run", "python", "run_pipeline.py"]
