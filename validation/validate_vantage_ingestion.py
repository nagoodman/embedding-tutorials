import sys
import json
import pandas as pd
import argparse
from typing import Dict, Any, List, Union
import re
from colorama import Fore, Back, Style, init
import os
from datetime import datetime
import io

# Initialize colorama for cross-platform colored output
init()


def format_file_info(file_path: str, df: pd.DataFrame) -> str:
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    file_type = os.path.splitext(file_path)[1].lstrip(".")
    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

    output = [
        f"\n{Fore.CYAN}{Style.BRIGHT}=== File Information ==={Style.RESET_ALL}",
        f"{Fore.GREEN}File:{Style.RESET_ALL} {os.path.basename(file_path)}",
        f"{Fore.GREEN}Type:{Style.RESET_ALL} {file_type.upper()}",
        f"{Fore.GREEN}Size:{Style.RESET_ALL} {file_size:.2f} MB",
        f"{Fore.GREEN}Last modified:{Style.RESET_ALL} {last_modified.strftime('%Y-%m-%d %H:%M:%S')}",
        f"{Fore.GREEN}Rows:{Style.RESET_ALL} {df.shape[0]}",
        f"{Fore.GREEN}Columns:{Style.RESET_ALL} {df.shape[1]}",
        f"\n{Fore.YELLOW}Column names:{Style.RESET_ALL}",
        ", ".join(df.columns),
        f"\n{Fore.YELLOW}Sample data (first 5 rows):{Style.RESET_ALL}",
    ]

    # Select a subset of columns to display
    display_columns = ["id", "text", "meta_sku"]
    sample_df = df[display_columns].head()

    # Truncate long text fields
    max_text_length = 50
    sample_df["text"] = sample_df["text"].apply(
        lambda x: (x[:max_text_length] + "...") if len(x) > max_text_length else x
    )

    # Convert DataFrame to string with custom formatting
    table_string = sample_df.to_string(
        index=False, justify="left", col_space=20, max_colwidth=20
    )

    # Add colors to the table header
    table_lines = table_string.split("\n")
    table_lines[0] = f"{Fore.CYAN}{table_lines[0]}{Style.RESET_ALL}"
    colored_table = "\n".join(table_lines)

    output.append(colored_table)

    return "\n".join(output)


def format_validation_results(results: Dict[str, Any]) -> str:
    """
    Format the validation results into a more readable and actionable output.

    Args:
        results (Dict[str, Any]): The validation results dictionary.

    Returns:
        str: A formatted string containing the validation results.
    """
    output = []

    # Overall summary
    total_checks = len(results["validation_results"])
    passed_checks = sum(
        1 for check in results["validation_results"].values() if check["pass"]
    )

    output.append(
        f"\n{Fore.CYAN}=== Vantage Ingestion Format Validation Report ==={Style.RESET_ALL}"
    )
    output.append(
        f"\nOverall Result: {Fore.GREEN if passed_checks == total_checks else Fore.RED}{passed_checks}/{total_checks} checks passed{Style.RESET_ALL}"
    )

    # Field information
    output.append(f"\n{Fore.CYAN}Field Information:{Style.RESET_ALL}")
    for field, type_ in results["field_info"].items():
        output.append(f"  {field}: {type_}")

    # Detailed results
    output.append(f"\n{Fore.CYAN}Validation Details:{Style.RESET_ALL}")
    for check, result in results["validation_results"].items():
        check_name = check.replace("_", " ").capitalize()
        if result["pass"]:
            output.append(f"  {Fore.GREEN}✓ {check_name}{Style.RESET_ALL}")
        else:
            output.append(f"  {Fore.RED}✗ {check_name}{Style.RESET_ALL}")
            for error in result["errors"]:
                output.append(f"    - {error}")

    # Recommendations
    if passed_checks < total_checks:
        output.append(f"\n{Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
        if not results["validation_results"]["id_uniqueness"]["pass"]:
            output.append("  - Ensure all 'id' values are unique across the dataset.")
        if not results["validation_results"]["text_not_null"]["pass"]:
            output.append("  - Make sure all 'text' fields have non-null values.")
        if not results["validation_results"]["embeddings_check"]["pass"]:
            output.append(
                "  - Verify that all 'embeddings' are valid arrays of the correct dimension."
            )
        if not results["validation_results"]["operation_check"]["pass"]:
            output.append(
                "  - Check that all 'operation' values are either 'delete', 'update', or 'add'."
            )
        if not results["validation_results"]["meta_fields_check"]["pass"]:
            output.append(
                "  - Review meta field names to ensure they follow the correct format."
            )
        if not results["validation_results"]["text_or_embeddings_present_check"][
            "pass"
        ]:
            output.append(
                "  - Ensure each row has either 'text' or 'embeddings', but not both."
            )

    return "\n".join(output)


def read_file(file_input: Union[str, io.IOBase]) -> pd.DataFrame:
    """
    Read a JSONL or Parquet file and return a pandas DataFrame.

    Args:
        file_input (Union[str, io.IOBase]): Path to the input file or a file-like object.

    Returns:
        pd.DataFrame: DataFrame containing the file contents.

    Raises:
        ValueError: If the file format is not supported.
    """
    if isinstance(file_input, str):
        # If file_input is a string, treat it as a file path
        file_path = file_input
        if file_path.endswith(".jsonl"):
            # Specify dtypes for known columns
            dtypes = {"id": str, "text": str}
            # Read the JSONL file with explicit dtypes
            df = pd.read_json(file_path, lines=True, dtype=dtypes)
        elif file_path.endswith(".parquet"):
            # Read the Parquet file
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(
                "Unsupported file format. Please use .jsonl or .parquet files."
            )
    elif isinstance(file_input, io.IOBase):
        # If file_input is a file-like object, try to read it as JSONL
        try:
            dtypes = {"id": str, "text": str}
            df = pd.read_json(file_input, lines=True, dtype=dtypes)
        except ValueError:
            # If JSONL reading fails, try reading as Parquet
            file_input.seek(0)  # Reset file pointer to the beginning
            try:
                df = pd.read_parquet(io.BytesIO(file_input.read()))
            except:
                raise ValueError("Unable to read input as JSONL or Parquet.")
    else:
        raise ValueError(
            "Invalid input type. Expected file path string or file-like object."
        )

    # Ensure 'id' and 'text' are treated as strings if they exist
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    if "text" in df.columns:
        df["text"] = df["text"].astype(str)

    return df


def validate_file(df: pd.DataFrame, dimension_size: int = None) -> Dict[str, Any]:
    """
    Validate the input DataFrame according to Vantage Ingestion Format specifications.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to validate.
        dimension_size (int, optional): Expected dimension size for embeddings.

    Returns:
        Dict[str, Any]: A dictionary containing validation results.
    """
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
            "delete_operation_validity_check": {"pass": True, "errors": []},
        },
    }

    results["field_info"] = df.dtypes.apply(lambda x: str(x)).to_dict()

    # Check if 'id' and 'text' fields exist and have string data type
    # Ref: Vantage Ingestion Docs - Required Fields
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
    # Ref: Vantage Ingestion Docs - Required Fields - 'id'
    if "id" in df.columns:
        # Check for null 'id' values
        null_ids = df[df["id"].isnull()]
        if not null_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            results["validation_results"]["id_uniqueness"]["errors"].extend(
                f"Null 'id' value found in row {index}" for index in null_ids.index
            )

        # Check for non-string 'id' values
        non_string_ids = df[~df["id"].apply(lambda x: isinstance(x, str))]
        if not non_string_ids.empty:
            results["validation_results"]["id_field_check"]["pass"] = False
            results["validation_results"]["id_field_check"]["errors"].extend(
                f"Non-string 'id' value found in row {index}: {value}"
                for index, value in non_string_ids["id"].items()
            )

        # Check 'id' length (1 to 256 characters)
        invalid_length_ids = df[~df["id"].apply(lambda x: 1 <= len(str(x)) <= 256)]
        if not invalid_length_ids.empty:
            results["validation_results"]["id_length_check"]["pass"] = False
            results["validation_results"]["id_length_check"]["errors"].extend(
                f"'id' field length not between 1 and 256 characters in row {index}"
                for index in invalid_length_ids.index
            )

    # Check 'text' field
    # Ref: Vantage Ingestion Docs - Required Fields - 'text'
    if "text" in df.columns:
        # Check for null 'text' values
        null_texts = df[df["text"].isnull()]
        if not null_texts.empty:
            results["validation_results"]["text_not_null"]["pass"] = False
            results["validation_results"]["text_not_null"]["errors"].extend(
                f"Null 'text' value found in row {index}" for index in null_texts.index
            )

        # Check for non-string 'text' values
        non_string_texts = df[~df["text"].apply(lambda x: isinstance(x, str))]
        if not non_string_texts.empty:
            results["validation_results"]["text_field_check"]["pass"] = False
            results["validation_results"]["text_field_check"]["errors"].extend(
                f"Non-string 'text' value found in row {index}: {value}"
                for index, value in non_string_texts["text"].items()
            )

        # Check 'text' size (max 1GB)
        large_texts = df[df["text"].apply(lambda x: len(str(x)) > 1e9)]
        if not large_texts.empty:
            results["validation_results"]["text_size_check"]["pass"] = False
            results["validation_results"]["text_size_check"]["errors"].extend(
                f"'text' field size exceeds 1GB in row {index}"
                for index in large_texts.index
            )

    # Check for text or embeddings present
    # Ref: Vantage Ingestion Docs - Required Fields
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

    # Check delete operation validity
    # Ref: Vantage Ingestion Docs - Optional Fields - 'operation'
    if "operation" in df.columns:
        delete_rows = df[df["operation"] == "delete"]
        if not delete_rows.empty:
            # Check if 'id' is present and not null for delete operations
            missing_id = delete_rows["id"].isnull()
            if missing_id.any():
                results["validation_results"]["delete_operation_validity_check"][
                    "pass"
                ] = False
                results["validation_results"]["delete_operation_validity_check"][
                    "errors"
                ].append(
                    f"'delete' operation rows missing 'id': {missing_id[missing_id].index.tolist()}"
                )

            # Check if any delete rows have 'text' or 'embeddings'
            invalid_delete_rows = (
                delete_rows[["text", "embeddings"]].notna().any(axis=1)
            )
            if invalid_delete_rows.any():
                results["validation_results"]["delete_operation_validity_check"][
                    "pass"
                ] = False
                results["validation_results"]["delete_operation_validity_check"][
                    "errors"
                ].append(
                    f"'delete' operation rows with 'text' or 'embeddings': {invalid_delete_rows[invalid_delete_rows].index.tolist()}"
                )

    # Check for duplicate 'id' values
    # Ref: Vantage Ingestion Docs - Required Fields - 'id'
    if "id" in df.columns:
        duplicate_ids = df[df["id"].duplicated(keep=False)]
        if not duplicate_ids.empty:
            results["validation_results"]["id_uniqueness"]["pass"] = False
            for id_value, group in duplicate_ids.groupby("id"):
                results["validation_results"]["id_uniqueness"]["errors"].append(
                    f"Duplicate 'id' value '{id_value}' found in rows: {group.index.tolist()}"
                )

    # Check 'embeddings' field (optional)
    # Ref: Vantage Ingestion Docs - Required Fields - 'embeddings'
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

    # Operation check
    # Ref: Vantage Ingestion Docs - Optional Fields - 'operation'
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

    # meta fields check
    # Ref: Vantage Ingestion Docs - Optional Fields - 'meta_' fields
    meta_pattern = re.compile(r"^meta_[a-zA-Z0-9_-]{3,255}$")
    meta_ordered_pattern = re.compile(r"^meta_ordered_[a-zA-Z0-9_-]{3,}$")
    meta_facet_pattern = re.compile(r"^meta_facet_[a-zA-Z0-9_-]{3,}$")

    for column in df.columns:
        if column not in ["id", "text", "embeddings", "operation"]:
            if not (
                meta_pattern.match(column)
                or meta_ordered_pattern.match(column)
                or meta_facet_pattern.match(column)
            ):
                results["validation_results"]["meta_fields_check"]["pass"] = False
                results["validation_results"]["meta_fields_check"]["errors"].append(
                    f"Invalid meta field name: '{column}'. Meta field names should start with 'meta_', 'meta_ordered_', or 'meta_facet_'."
                )
            elif column.startswith("meta_"):
                fieldname = column.split("_", 1)[1]
                if len(fieldname) < 3:
                    results["validation_results"]["meta_fields_check"]["pass"] = False
                    results["validation_results"]["meta_fields_check"]["errors"].append(
                        f"Meta field name '{column}' is too short. Minimum length is 3 characters after 'meta_'."
                    )

    return results


def main(file_input: Union[str, io.IOBase], dimension_size: int = None):
    try:
        df = read_file(file_input)
        print(f"File read successfully. Shape: {df.shape}")
        results = validate_file(df, dimension_size)
        formatted_results = format_validation_results(results)
        print(formatted_results)
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
        help="Expected dimension size for embeddings if included in the dataset (optional). "
        "If provided, embeddings will be validated against this size. "
        "The 'embeddings' field itself is optional.",
    )
    args = parser.parse_args()

    main(args.file_path, args.dimension_size)
