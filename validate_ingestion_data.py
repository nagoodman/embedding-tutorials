import sys
import json
import pandas as pd
import argparse
from typing import Dict, Any

def read_file(file_path: str) -> pd.DataFrame:
    """
    Read a JSONL or Parquet file and return a pandas DataFrame.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        pd.DataFrame: DataFrame containing the file contents.
    
    Raises:
        ValueError: If the file format is not supported.
    """
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

    # Validate 'id' uniqueness and non-null
    if 'id' in df.columns:
        null_ids = df[df['id'].isnull()]
        if not null_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            results["validation_results"]["id_uniqueness"]["errors"].append(f"Null 'id' values found in rows: {null_ids.index.tolist()}")

        duplicate_ids = df[df['id'].duplicated()]
        if not duplicate_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            results["validation_results"]["id_uniqueness"]["errors"].append(f"Duplicate 'id' values found: {duplicate_ids['id'].tolist()}")

    # Validate 'text' non-null
    if 'text' in df.columns:
        null_texts = df[df['text'].isnull()]
        if not null_texts.empty:
            results["validation_results"]["text_not_null"]["pass"] = False
            results["validation_results"]["text_not_null"]["errors"].append(f"Null 'text' values found in rows: {null_texts.index.tolist()}")

    return results

def main(file_path: str):
    """
    Main function to read the file, perform validations, and print results.
    
    Args:
        file_path (str): Path to the input file.
    """
    try:
        # Read the file
        df = read_file(file_path)
        
        # Perform validations
        results = validate_file(df)

        # Print results
        print("File validation results:")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validate JSONL or Parquet files for specific criteria.")
    parser.add_argument("file_path", help="Path to the JSONL or Parquet file to validate")
    args = parser.parse_args()

    # Run the main function with the provided file path
    main(args.file_path)
