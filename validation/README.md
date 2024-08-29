<p align="center">
  <img src="./assets/logo.svg" alt="Vantage Ingestion Validator Logo" width="200"/>
</p>

# Vantage Ingestion Validator

# Vantage Ingestion Validator

## Overview

The Vantage Ingestion Validator is a Python script designed to validate JSONL and Parquet files according to the Vantage Ingestion Format (VIF) specifications. It ensures that data files meet the required structure and content guidelines before ingestion into the Vantage system.

## Features

- Supports both JSONL and Parquet file formats
- Validates presence and data types of required fields ('id', 'text', 'embeddings')
- Checks for field uniqueness, non-null values, and size constraints
- Validates optional fields and metadata
- Provides detailed error reporting for each validation check
- Offers a user-friendly, color-coded output for easy interpretation of results

## Project Structure

```
vantage-ingestion-validator/
│
├── validate_vantage_ingestion.py
├── README.md
├── requirements.txt
│
└── tests/
    ├── __init__.py
    ├── test_validate_vantage_ingestion.py
    └── test_data/
        ├── valid_sample.jsonl
        ├── valid_sample.parquet
        ├── invalid_sample.jsonl
        └── invalid_sample.parquet
```

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the script from the command line:

```
python validate_vantage_ingestion.py <file_path> [--dimension_size <size>]
```

Arguments:

- `<file_path>`: Path to the JSONL or Parquet file to validate
- `--dimension_size`: (Optional) Expected dimension size for embeddings

Example:

```
python validate_vantage_ingestion.py data.jsonl --dimension_size 1536
```

## Output

The script provides a color-coded, user-friendly output that includes:

1. File Information:
   - File name, type, and size
   - Last modified date
   - Number of rows and columns
   - List of column names
   - Sample of the first 5 rows (truncated for readability)

2. Validation Results:
   - Overall summary of passed checks
   - Detailed results for each validation check
   - Specific error messages for failed checks
   - Recommendations for fixing issues when checks fail

## Validation Checks

The validator performs the following checks:

1. Field presence and data type validation
2. 'id' uniqueness and non-null check
3. 'text' non-null check
4. 'embeddings' format and dimension size check (if applicable)
5. 'operation' field validation
6. Metadata field naming and content validation
7. File size and content length checks

## Extending Validation Rules

To extend the validator with new rules:

1. Add new validation checks in the `validate_file` function in `validate_vantage_ingestion.py`
2. Update the `results` dictionary to include new validation categories
