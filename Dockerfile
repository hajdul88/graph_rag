FROM python:3.9-slim

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

ENV NEO4J_URL=""
ENV NEO4J_USER=""
ENV NEO4J_PASSWORD=""
ENV HF_API_TOKEN=""


CMD ["poetry", "run", "python", "main.py"]
