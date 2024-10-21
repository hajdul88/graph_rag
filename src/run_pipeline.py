import os
import asyncio
from directory_reader import DirectoryFileReader

async def main():
    """
    Main function to run the DirectoryFileReader and output chunked content
    of .txt and .pdf files.
    """

    directory = "../example_data"

    mode = "separator"
    overlap_size = 0
    txt_separator = "\n\n"  # Example separator for text files (line break separator)

    directory_reader = DirectoryFileReader(directory, mode=mode, chunk_size=500, txt_separator=txt_separator, overlap_size=overlap_size)

    file_chunks = []

    async for chunk in directory_reader.read_files():
        if chunk.strip():
            file_chunks.append(chunk)

    print("All file chunks stored in memory:")
    for index, chunk in enumerate(file_chunks):
        print(f"Chunk {index + 1}:\n{chunk}\n{'-' * 80}")

if __name__ == "__main__":
    asyncio.run(main())