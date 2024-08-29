import sys
import json
import pandas as pd
import argparse
from typing import Dict, Any, List
import re


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
    if file_path.endswith(".jsonl"):
        # Specify dtypes for known columns
        dtypes = {"id": str, "text": str}

        # Read the JSONL file with explicit dtypes
        df = pd.read_json(file_path, lines=True, dtype=dtypes)

        # Ensure 'id' and 'text' are treated as strings
        df["id"] = df["id"].astype(str)
        df["text"] = df["text"].astype(str)

        return df

    elif file_path.endswith(".parquet"):
        # Read the Parquet file
        df = pd.read_parquet(file_path)

        # Ensure 'id' and 'text' are treated as strings if they exist
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        if "text" in df.columns:
            df["text"] = df["text"].astype(str)

        return df
    else:
        raise ValueError(
            "Unsupported file format. Please use .jsonl or .parquet files."
        )


def validate_file(df: pd.DataFrame, dimension_size: int = None) -> Dict[str, Any]:
    results = {
        "field_info": {},
        "validation_results": {
            "id_field_check": {"pass": True, "errors": []},
            "text_field_check": {"pass": True, "errors": []},
            "id_uniqueness": {"pass": True, "errors": []},
            "text_not_null": {"pass": True, "errors": []},
            "embeddings_check": {"pass": True, "errors": []},
            "operation_check": {"pass": True, "errors": []},
            "meta_fields_check": {"pass": True, "errors": []},
            "meta_ordered_fields_check": {"pass": True, "errors": []},
            "meta_facet_fields_check": {"pass": True, "errors": []},
            "id_length_check": {"pass": True, "errors": []},
            "text_size_check": {"pass": True, "errors": []},
            "text_or_embeddings_present_check": {"pass": True, "errors": []},
            "delete_operation_check": {"pass": True, "errors": []},
        },
    }

    results["field_info"] = df.dtypes.apply(lambda x: str(x)).to_dict()

    # Check if 'id' and 'text' fields exist and have string data type
    if "id" not in df.columns or df["id"].dtype != "object":
        results["validation_results"]["id_field_check"]["pass"] = False
        results["validation_results"]["id_field_check"]["errors"].append(
            "'id' field is missing or not of string type"
        )

    if "text" not in df.columns or df["text"].dtype != "object":
        results["validation_results"]["text_field_check"]["pass"] = False
        results["validation_results"]["text_field_check"]["errors"].append(
            "'text' field is missing or not of string type"
        )

    # Check 'id' field
    if "id" in df.columns:
        null_ids = df[df["id"].isnull()]
        if not null_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            results["validation_results"]["id_uniqueness"]["errors"].extend(
                f"Null 'id' value found in row {index}" for index in null_ids.index
            )

        non_string_ids = df[~df["id"].apply(lambda x: isinstance(x, str))]
        if not non_string_ids.empty:
            results["validation_results"]["id_field_check"]["pass"] = False
            results["validation_results"]["id_field_check"]["errors"].extend(
                f"Non-string 'id' value found in row {index}: {value}"
                for index, value in non_string_ids["id"].items()
            )

        invalid_length_ids = df[~df["id"].apply(lambda x: 1 <= len(str(x)) <= 256)]
        if not invalid_length_ids.empty:
            results["validation_results"]["id_length_check"]["pass"] = False
            results["validation_results"]["id_length_check"]["errors"].extend(
                f"'id' field length not between 1 and 256 characters in row {index}"
                for index in invalid_length_ids.index
            )

    # Check 'text' field
    if "text" in df.columns:
        null_texts = df[df["text"].isnull()]
        if not null_texts.empty:
            results["validation_results"]["text_not_null"]["pass"] = False
            results["validation_results"]["text_not_null"]["errors"].extend(
                f"Null 'text' value found in row {index}" for index in null_texts.index
            )

        non_string_texts = df[~df["text"].apply(lambda x: isinstance(x, str))]
        if not non_string_texts.empty:
            results["validation_results"]["text_field_check"]["pass"] = False
            results["validation_results"]["text_field_check"]["errors"].extend(
                f"Non-string 'text' value found in row {index}: {value}"
                for index, value in non_string_texts["text"].items()
            )

        large_texts = df[df["text"].apply(lambda x: len(str(x)) > 1e9)]
        if not large_texts.empty:
            results["validation_results"]["text_size_check"]["pass"] = False
            results["validation_results"]["text_size_check"]["errors"].extend(
                f"'text' field size exceeds 1GB in row {index}"
                for index in large_texts.index
            )

    # Check for text or embeddings present
    text_present = (
        df["text"].notna() if "text" in df.columns else pd.Series([False] * len(df))
    )
    embeddings_present = (
        df["embeddings"].notna()
        if "embeddings" in df.columns
        else pd.Series([False] * len(df))
    )
    invalid_rows = ~(text_present ^ embeddings_present)
    if invalid_rows.any():
        results["validation_results"]["text_or_embeddings_present_check"][
            "pass"
        ] = False
        results["validation_results"]["text_or_embeddings_present_check"][
            "errors"
        ].extend(
            f"Row {index} does not have exactly one of 'text' or 'embeddings'"
            for index in df[invalid_rows].index
        )

    # Check delete operation
    if "operation" in df.columns:
        delete_rows = df[df["operation"] == "delete"]
        invalid_delete_rows = (
            delete_rows[delete_rows.columns.difference(["id", "operation"])]
            .notna()
            .any(axis=1)
        )
        if invalid_delete_rows.any():
            results["validation_results"]["delete_operation_check"]["pass"] = False
            results["validation_results"]["delete_operation_check"]["errors"].extend(
                f"'delete' operation row {index} has fields other than 'id'"
                for index in delete_rows[invalid_delete_rows].index
            )

    # Check for duplicate 'id' values
    if "id" in df.columns:
        duplicate_ids = df[df["id"].duplicated(keep=False)]
        if not duplicate_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            for id_value, group in duplicate_ids.groupby("id"):
                results["validation_results"]["id_uniqueness"]["errors"].append(
                    f"Duplicate 'id' value '{id_value}' found in rows: {group.index.tolist()}"
                )

    # Check 'embeddings' field (optional)
    if "embeddings" in df.columns:

        def validate_embedding(emb):
            if pd.isna(emb):  # Allow null values in embeddings column
                return True
            if not isinstance(emb, list):
                return False
            if dimension_size and len(emb) != dimension_size:
                return False
            return all(isinstance(i, float) for i in emb)

        invalid_embeddings = df[~df["embeddings"].apply(validate_embedding)]
        if not invalid_embeddings.empty:
            results["validation_results"]["embeddings_check"]["pass"] = False
            results["validation_results"]["embeddings_check"]["errors"].append(
                f"Invalid 'embeddings' found in rows: {invalid_embeddings.index.tolist()}. "
                f"Expected an array of {dimension_size} 32-bit floats or null."
            )

    # Check for text XOR embeddings (both are optional, but at least one should be present)
    text_present = (
        df["text"].notna() if "text" in df.columns else pd.Series([False] * len(df))
    )
    embeddings_present = (
        df["embeddings"].notna()
        if "embeddings" in df.columns
        else pd.Series([False] * len(df))
    )
    invalid_rows = ~(
        text_present | embeddings_present
    )  # Changed from XOR (^) to OR (|)
    if invalid_rows.any():
        results["validation_results"]["text_or_embeddings_present_check"][
            "pass"
        ] = False
        results["validation_results"]["text_or_embeddings_present_check"][
            "errors"
        ].extend(
            f"Row {index} must have at least one of 'text' or 'embeddings'"
            for index in df[invalid_rows].index
        )

    # Operation check
    if "operation" in df.columns:
        valid_operations = {"delete", "update", "add"}
        invalid_ops = df[~df["operation"].isin(valid_operations)]
        if not invalid_ops.empty:
            results["validation_results"]["operation_check"]["pass"] = False
            results["validation_results"]["operation_check"]["errors"].append(
                f"Invalid 'operation' values found in rows: {invalid_ops.index.tolist()}"
            )
    else:
        # If 'operation' is not specified, assume all operations are 'update'
        df["operation"] = "update"

    # Enhanced meta fields check
    meta_pattern = re.compile(r"^meta_[a-zA-Z0-9_-]{3,255}$")
    meta_ordered_pattern = re.compile(r"^meta_ordered_[a-zA-Z0-9_-]{3,}$")
    meta_facet_pattern = re.compile(r"^meta_facet_[a-zA-Z0-9_-]{3,}$")

    for column in df.columns:
        if column.startswith("meta_"):
            if (
                meta_pattern.match(column)
                or meta_ordered_pattern.match(column)
                or meta_facet_pattern.match(column)
            ):
                fieldname = column.split("_", 1)[1]
                if len(fieldname) < 3:
                    results["validation_results"]["meta_fields_check"]["pass"] = False
                    results["validation_results"]["meta_fields_check"]["errors"].append(
                        f"Meta field name '{column}' is too short. Minimum length is 3 characters after 'meta_'."
                    )

                # Existing type checks...
            else:
                results["validation_results"]["meta_fields_check"]["pass"] = False
                results["validation_results"]["meta_fields_check"]["errors"].append(
                    f"Invalid meta field name: '{column}'"
                )

    # Strict check for text XOR embeddings
    text_present = (
        df["text"].notna() if "text" in df.columns else pd.Series([False] * len(df))
    )
    embeddings_present = (
        df["embeddings"].notna()
        if "embeddings" in df.columns
        else pd.Series([False] * len(df))
    )
    invalid_rows = ~(text_present ^ embeddings_present)
    if invalid_rows.any():
        results["validation_results"]["text_or_embeddings_present_check"][
            "pass"
        ] = False
        results["validation_results"]["text_or_embeddings_present_check"][
            "errors"
        ].extend(
            f"Row {index} must have exactly one of 'text' or 'embeddings'"
            for index in df[invalid_rows].index
        )

    return results


def main(file_path: str, dimension_size: int = None):
    try:
        df = read_file(file_path)
        print(f"File read successfully. Shape: {df.shape}")
        results = validate_file(df, dimension_size)

        print("File validation results:")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate JSONL or Parquet files according to Vantage Ingestion Format."
    )
    parser.add_argument(
        "file_path", help="Path to the JSONL or Parquet file to validate"
    )
    parser.add_argument(
        "--dimension_size",
        type=int,
        help="Expected dimension size for embeddings (optional). "
        "If provided, embeddings will be validated against this size. "
        "The 'embeddings' field itself is optional.",
    )
    args = parser.parse_args()

    main(args.file_path, args.dimension_size)
