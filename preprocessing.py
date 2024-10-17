import pandas as pd
import os
from format_java import format_java_code


columns_to_keep = ['code', 'formatted']

def process_files_in_directory(directory_path):
    combined_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)
            data = load_data(file_path)
            
            if 'code' in data.columns:
                data['formatted'] = data['code'].apply(format_java_code)
                combined_data.append(data)
            else:
                print(f"Warning: 'code' column not found in {filename}")
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        return combined_df
    else:
        print("No data found to combine.")
        return pd.DataFrame()

def load_data(file_path):
    df = pd.read_json(file_path, lines=True)
    return df[columns_to_keep].head(10000)
