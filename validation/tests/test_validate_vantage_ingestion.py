import unittest
import pandas as pd
from io import StringIO
from validate_vantage_ingestion import read_file, validate_file


class TestVantageIngestionValidator(unittest.TestCase):

    def test_read_file_jsonl(self):
        # Test reading a valid JSONL file
        data = StringIO(
            '{"id": "1", "text": "Sample text"}\n{"id": "2", "text": "Another sample"}'
        )
        df = read_file(data)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["id", "text"])

    def test_validate_file_valid(self):
        # Test validating a valid DataFrame
        data = {
            "id": ["1", "2"],
            "text": ["Sample text", "Another sample"],
            "meta_category": ["A", "B"],
        }
        df = pd.DataFrame(data)
        results = validate_file(df)
        self.assertTrue(
            all(result["pass"] for result in results["validation_results"].values())
        )

    def test_validate_file_invalid_meta(self):
        # Test validating a DataFrame with invalid metadata field name
        data = {
            "id": ["1", "2"],
            "text": ["Sample text", "Another sample"],
            "invalid_meta": ["A", "B"],
        }
        df = pd.DataFrame(data)
        results = validate_file(df)
        self.assertFalse(results["validation_results"]["meta_fields_check"]["pass"])

    def test_validate_file_missing_text(self):
        # Test validating a DataFrame with missing text
        data = {"id": ["1", "2"], "text": ["Sample text", None]}
        df = pd.DataFrame(data)
        results = validate_file(df)
        self.assertFalse(results["validation_results"]["text_not_null"]["pass"])

    def test_validate_file_invalid_meta(self):
        # Test validating a DataFrame with invalid metadata field name
        data = {
            "id": ["1", "2"],
            "text": ["Sample text", "Another sample"],
            "meta_valid": ["A", "B"],
            "invalid_meta": ["C", "D"],
        }
        df = pd.DataFrame(data)
        results = validate_file(df)
        self.assertFalse(results["validation_results"]["meta_fields_check"]["pass"])
        self.assertIn(
            "Invalid meta field name: 'invalid_meta'",
            results["validation_results"]["meta_fields_check"]["errors"][0],
        )


if __name__ == "__main__":
    unittest.main()
