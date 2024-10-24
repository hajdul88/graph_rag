import os
import time
import asyncio
from directory_reader import DirectoryFileReader
from chunk_to_network import neo4j_processor as neo4j_ingestion


# This is only here for the local setup, otherwise its not needed.
def init():
    """
    Initialize environment variables for all parameters.
    """
    os.environ['directory'] = "../example_data"
    os.environ['mode'] = "size"
    os.environ['chunk_size'] = "300"
    os.environ['overlap_size'] = "20"
    os.environ['txt_separator'] = "\n\n"
    os.environ['NEO4J_URI'] = "bolt://localhost:7687"
    os.environ['NEO4J_USER'] = "neo4j"
    os.environ['NEO4J_PASSWORD'] = "test1234"

async def main():
    """
    Main function to run the DirectoryFileReader and output chunked content
    of .txt and .pdf files.
    """

    ## PARAMETERS
    directory = os.environ['directory']
    mode = os.environ['mode']
    chunk_size = int(os.environ['chunk_size'])
    overlap_size = int(os.environ['overlap_size'])
    txt_separator = os.environ['txt_separator']
    NEO4J_URI = os.environ['NEO4J_URI']
    NEO4J_USER = os.environ['NEO4J_USER']
    NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']

    directory_reader = DirectoryFileReader(directory, mode=mode, chunk_size=chunk_size, txt_separator=txt_separator, overlap_size=overlap_size)

    chunk_processor = neo4j_ingestion.ChunkProcessor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    await chunk_processor.process_chunks(directory_reader)

    chunk_processor.close()


if __name__ == "__main__":

    #init() # To run locally
    asyncio.run(main())
    time.sleep(20)