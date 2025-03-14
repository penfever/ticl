"""
Tests for semantic data loading functionality.
"""

import unittest
import torch
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from ticl.datasets.semantic_prior_data_loader import (
    load_semantic_prior_data,
    get_random_semantic_data,
    is_numeric_column
)

from ticl.datasets.labeled_numeric_prior_data_loader import (
    load_numeric_prior_data,
    extract_numeric_values
)

class TestSemanticDataLoading(unittest.TestCase):
    """Test the semantic data loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary JSON file with mock column data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file_path = os.path.join(self.temp_dir.name, "test_columns.json")
        
        # Create sample column data
        self.test_data = {
            "name": [1, 2, 3, 4, 5] * 20,
            "age": [10, 20, 30, 40, 50] * 20,
            "income": [1000, 2000, 3000, 4000, 5000] * 20,
            "city": [10, 20, 30, 40, 50] * 20,
            "country": [100, 200, 300, 400, 500] * 20
        }
        
        # Write to file
        with open(self.log_file_path, 'w') as f:
            json.dump(self.test_data, f)
        
        # Mock the CLIPTokenizer
        self.tokenizer_patcher = patch('transformers.CLIPTokenizerFast.from_pretrained')
        self.mock_tokenizer = self.tokenizer_patcher.start()
        
        # Set up mock tokenizer behavior
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.ones(1, 10)
        }
        self.mock_tokenizer.return_value = mock_tokenizer_instance
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
        self.tokenizer_patcher.stop()
    
    def test_is_numeric_column(self):
        """Test the numeric column detection function."""
        numeric_columns = [
            "age", "count", "price", "income", "year", 
            "weight_kg", "height_cm", "temperature", "distance"
        ]
        
        non_numeric_columns = [
            "name", "city", "country", "category", "description",
            "title", "address", "email", "phone", "url"
        ]
        
        for col in numeric_columns:
            self.assertTrue(is_numeric_column(col), f"Column '{col}' should be detected as numeric")
            
        for col in non_numeric_columns:
            self.assertFalse(is_numeric_column(col), f"Column '{col}' should not be detected as numeric")
    
    def test_extract_numeric_values(self):
        """Test numeric value extraction from tokenized data."""
        test_cases = [
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            (["10", "20", "30"], [10, 20, 30]),
            (["1.5", "2.5", "3.5"], [1.5, 2.5, 3.5]),
            (["value", "10", "name", "20"], [10, 20]),
            (["mixed_123.45_data"], [123.45]),
            (["1e3", "2e-2"], [1000, 0.02])
        ]
        
        for input_data, expected in test_cases:
            result = extract_numeric_values(input_data)
            self.assertEqual(len(result), len(expected), 
                             f"Expected {len(expected)} values, got {len(result)}")
            
            for r, e in zip(result, expected):
                self.assertAlmostEqual(r, e, places=5, 
                                      msg=f"Expected {e}, got {r}")
    
    def test_load_semantic_prior_data(self):
        """Test loading semantic prior data."""
        # Mock the column name detection to ensure consistent behavior
        with patch('ticl.datasets.semantic_prior_data_loader.is_numeric_column',
                  side_effect=lambda x: x in ['age', 'income']):
            
            # Test with force_non_numeric=True
            name_tokens, value_tokens = load_semantic_prior_data(
                log_file_path=self.log_file_path,
                force_non_numeric=True,
                target_tensor_size=150
            )
            
            # Check that only non-numeric columns are included
            self.assertIn('name', name_tokens)
            self.assertIn('city', name_tokens)
            self.assertIn('country', name_tokens)
            self.assertNotIn('age', name_tokens)
            self.assertNotIn('income', name_tokens)
            
            # Check tensor sizes
            for tensor in value_tokens.values():
                self.assertEqual(tensor.size(0), 150)
            
            # Test with force_non_numeric=False
            name_tokens, value_tokens = load_semantic_prior_data(
                log_file_path=self.log_file_path,
                force_non_numeric=False,
                target_tensor_size=200
            )
            
            # Check that all columns are included
            self.assertIn('name', name_tokens)
            self.assertIn('city', name_tokens)
            self.assertIn('country', name_tokens)
            self.assertIn('age', name_tokens)
            self.assertIn('income', name_tokens)
            
            # Check tensor sizes
            for tensor in value_tokens.values():
                self.assertEqual(tensor.size(0), 200)
    
    def test_load_numeric_prior_data(self):
        """Test loading numeric prior data."""
        # Mock the column name detection to ensure consistent behavior
        with patch('ticl.datasets.labeled_numeric_prior_data_loader.is_numeric_column',
                  side_effect=lambda x: x in ['age', 'income']):
            
            # Mock extract_numeric_values to return consistent data
            with patch('ticl.datasets.labeled_numeric_prior_data_loader.extract_numeric_values',
                      side_effect=lambda x: [float(i) for i in range(10)] if len(x) > 0 else []):
                
                # Test with force_numeric=True
                numeric_data = load_numeric_prior_data(
                    log_file_path=self.log_file_path,
                    force_numeric=True
                )
                
                # Check that only numeric columns are included
                self.assertIn('age', numeric_data)
                self.assertIn('income', numeric_data)
                self.assertNotIn('name', numeric_data)
                self.assertNotIn('city', numeric_data)
                self.assertNotIn('country', numeric_data)
                
                # Test with force_numeric=False
                numeric_data = load_numeric_prior_data(
                    log_file_path=self.log_file_path,
                    force_numeric=False
                )
                
                # Check that all columns with numeric values are included
                self.assertIn('age', numeric_data)
                self.assertIn('income', numeric_data)
                self.assertIn('name', numeric_data)
                self.assertIn('city', numeric_data)
                self.assertIn('country', numeric_data)
    
    def test_get_random_semantic_data(self):
        """Test getting random semantic data."""
        # Test with non-existent file (should generate synthetic data)
        random_data = get_random_semantic_data(
            num_classes=3,
            num_tokens=50,
            tensor_size=100,
            log_file_path="nonexistent_file.json"
        )
        
        self.assertEqual(random_data.shape, (3, 100))
        self.assertTrue(torch.all(random_data >= 1))
        self.assertTrue(torch.all(random_data <= 49404))
        
        # Test with real file
        with patch('ticl.datasets.semantic_prior_data_loader.load_semantic_prior_data',
                  return_value=({}, {k: torch.ones(200) for k in ['col1', 'col2', 'col3', 'col4']})):
            
            random_data = get_random_semantic_data(
                num_classes=2,
                tensor_size=200,
                log_file_path=self.log_file_path
            )
            
            self.assertEqual(random_data.shape, (2, 200))


if __name__ == '__main__':
    unittest.main()