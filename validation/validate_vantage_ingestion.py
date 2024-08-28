import sys
import json
import pandas as pd
import argparse
from typing import Dict, Any

def read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.jsonl'):
        return pd.read_json(file_path, lines=True)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .jsonl or .parquet files.")

def validate_file(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the contents of the DataFrame according to specified rules.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
    
    Returns:
        Dict[str, Any]: Dictionary containing validation results.
    """
    results = {
        "field_info": {},
        "validation_results": {
            "id_field_check": {"pass": True, "errors": []},
            "text_field_check": {"pass": True, "errors": []},
            "id_uniqueness": {"pass": True, "errors": []},
            "text_not_null": {"pass": True, "errors": []}
        }
    }

    # Output fields and data types
    results["field_info"] = df.dtypes.apply(lambda x: str(x)).to_dict()

    # Check if 'id' and 'text' fields exist and have string data type
    if 'id' not in df.columns or df['id'].dtype != 'object':
        results["validation_results"]["id_field_check"]["pass"] = False
        results["validation_results"]["id_field_check"]["errors"].append("'id' field is missing or not of string type")
    
    if 'text' not in df.columns or df['text'].dtype != 'object':
        results["validation_results"]["text_field_check"]["pass"] = False
        results["validation_results"]["text_field_check"]["errors"].append("'text' field is missing or not of string type")

    # Validate each row
    for index, row in df.iterrows():
        # Check 'id' field
        if 'id' in df.columns:
            if pd.isna(row['id']):
                results["validation_results"]["id_uniqueness"]["pass"] = False
                results["validation_results"]["id_uniqueness"]["errors"].append(f"Null 'id' value found in row {index}")
            elif not isinstance(row['id'], str):
                results["validation_results"]["id_field_check"]["pass"] = False
                results["validation_results"]["id_field_check"]["errors"].append(f"Non-string 'id' value found in row {index}: {row['id']}")

        # Check 'text' field
        if 'text' in df.columns:
            if pd.isna(row['text']):
                results["validation_results"]["text_not_null"]["pass"] = False
                results["validation_results"]["text_not_null"]["errors"].append(f"Null 'text' value found in row {index}")
            elif not isinstance(row['text'], str):
                results["validation_results"]["text_field_check"]["pass"] = False
                results["validation_results"]["text_field_check"]["errors"].append(f"Non-string 'text' value found in row {index}: {row['text']}")

    # Check for duplicate 'id' values
    if 'id' in df.columns:
        duplicate_ids = df[df['id'].duplicated(keep=False)]
        if not duplicate_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            for id_value, group in duplicate_ids.groupby('id'):
                results["validation_results"]["id_uniqueness"]["errors"].append(f"Duplicate 'id' value '{id_value}' found in rows: {group.index.tolist()}")

    return results

def main(file_path: str):
    try:
        df = read_file(file_path)
        results = validate_file(df)

        print("File validation results:")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        main(sys.argv[1])
