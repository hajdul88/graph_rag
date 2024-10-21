import pytest
import asyncio
from src.file_readers.text_reader import TextFileReader


@pytest.mark.asyncio
async def test_text_file_reader():
    reader = TextFileReader()
    content = ""

    async for chunk in reader.read("tests/test_data/doc_1.txt"):
        content += chunk

    assert "climate change presents" in content

