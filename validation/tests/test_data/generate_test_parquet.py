import pandas as pd

# Generate valid Parquet file
valid_data = {
    "id": ["1", "2", "3"],
    "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
    "meta_category": ["A", "B", "A"],
    "meta_score": [0.95, 0.87, 0.92],
}
valid_df = pd.DataFrame(valid_data)
valid_df.to_parquet("./test_data/valid_sample.parquet", index=False)

# Generate invalid Parquet file
invalid_data = {
    "id": ["1", "1", "3", "4"],
    "text": ["Sample text 1", "Duplicate ID", None, "Missing metadata"],
    "meta_category": ["A", "B", "A", "C"],
    "invalid_meta": ["Invalid", "metadata", "field", "name"],
}
invalid_df = pd.DataFrame(invalid_data)
invalid_df.to_parquet("./test_data/invalid_sample.parquet", index=False)

print("Test Parquet files generated successfully.")
