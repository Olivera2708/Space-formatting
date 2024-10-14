import preprocessing
from format_java import format_java_code
import pandas as pd

def create_datasets():
    # data = preprocessing.process_files_in_directory("data/train")
    # data['formatted'] = data['code'].apply(format_java_code)
    # data.to_json("training.jsonl", orient="records", lines=True)

    # data = preprocessing.load_data("data/test_orig.jsonl")
    # data['formatted'] = data['code'].apply(format_java_code)
    # data.to_json("test.jsonl", orient="records", lines=True)

    data = preprocessing.load_data("data/valid_orig.jsonl")
    data['formatted'] = data['code'].apply(format_java_code)
    data = data.dropna(subset=['formatted'])
    data = data[data['formatted'] != data['code']]
    data.to_json("data/valid.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    create_datasets()
