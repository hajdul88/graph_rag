import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))