# Use the official Python image from Docker Hub as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the pyproject.toml and poetry.lock to the working directory inside the container
COPY pyproject.toml poetry.lock /app/

# Install Poetry (a dependency manager for Python)
RUN pip install poetry

# Install the dependencies listed in pyproject.toml
RUN poetry install --no-dev

# Copy the rest of the application code to the container
COPY ./src /app/src
COPY ./tests /app/tests
COPY README.md /app/

# Set environment variables (optional)
ENV NEO4J_URI=bolt://neo4j:7687
ENV NEO4J_USER=neo4j
ENV NEO4J_PASSWORD=test

# Define the command that runs your application when the container starts
CMD ["poetry", "run", "python", "src/run_pipeline.py"]


