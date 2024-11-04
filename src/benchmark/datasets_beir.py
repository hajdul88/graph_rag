import os
from beir import util
import pandas as pd

def download_and_process_data(dataset_name: str, dataset_out_dir: str, files_location: str):
    """
        Downloads and processes datasets from BEIR benchmark collection.

        Processing steps:
        1. Downloads dataset from BEIR repository
        2. Extracts and processes corpus documents
        3. Chunks text into manageable sizes
        4. Saves processed files with titles and content

        Args:
            dataset_name (str): Name of the dataset to download from BEIR collection
            dataset_out_dir (str): Directory where the downloaded dataset will be stored
            files_location (str): Directory where processed files will be saved

        Returns:
            None

        Raises:
            ValueError: If dataset_name is not in the list of valid datasets

        Example:
            download_and_process_data("trec-covid", "./datasets", "./processed_files")
        """
    # List of valid dataset names from BEIR benchmark
    valid_names = ["trec-covid", "nfcorpus", "bioasq", "nq", "hotpotqa", "fiqa", "signal1m", "trec-news", "arguana",
                   "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs", "fever", "climate-fever",
                   "scifact"]
    # Validate dataset name
    if dataset_name not in valid_names:
        raise ValueError("Dataset name is not valid")

    # Construct and download from BEIR URL
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    data_path = util.download_and_unzip(url, dataset_out_dir)
    print("Benchmark dataset downloaded: {}".format(data_path))
    # Load document corpus into pandas DataFrame
    corpus = pd.read_json(f"{dataset_out_dir}/{dataset_name}/corpus.jsonl", lines=True)
    # Create output directory if needed
    if not os.path.isdir(f"{files_location}/{dataset_name}"):
        os.mkdir(f"{files_location}/{dataset_name}")

    # Process each document
    for index, row in corpus.iterrows():
        file_name = f"{files_location}/{dataset_name}/{row['_id']}.txt"  # Using `_id` as the filename
        text_words = row['text'].split()
        chunks = [text_words[i:i + 20] for i in range(0, len(text_words), 20)]
        with open(file_name, "w") as file:
            file.write(f"###{row['title']}\n\n")  # Write title
            for chunk in chunks:
                chunk_text = ' '.join(chunk)
                file.write(f"{chunk_text}")
