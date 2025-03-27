import os
import torch
import unittest
import random
import tempfile
import json
import time
import numpy as np
from unittest.mock import patch, MagicMock

# Add path to find ticl modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ticl.priors.classification_adapter import ClassificationAdapter
from ticl.priors.prior_bag import PriorBag
from ticl.datasets.semantic_prior_data_loader import load_semantic_prior_data, get_random_semantic_data


class TestSemanticPrior(unittest.TestCase):
    def setUp(self):
        """Setup test environment with mock data"""
        # Set random seeds for reproducibility
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Default device
        self.device = torch.device('cpu')
        
        # Create temporary JSON file with semantic data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_json_path = os.path.join(self.temp_dir.name, "test_columns.json")
        
        # Create mock semantic data
        mock_data = {}
        for i in range(10):
            column_name = f"test_column_{i}"
            # Create random token IDs between 1000-2000 for each column
            token_ids = [random.randint(1000, 2000) for _ in range(50)]
            mock_data[column_name] = token_ids
            
        # Write to temp file
        with open(self.temp_json_path, 'w') as f:
            json.dump(mock_data, f)

        # Mock prior configuration
        self.prior_config = {
            "feature_curriculum": False,
            "pad_zeros": False,
            "balanced": False,
            "multiclass_type": "rank", 
            "output_multiclass_ordered_p": 0.5,
            "multiclass_max_steps": 5,
            "num_classes": 5,
            "nan_prob_a_reason": 0.0,
            "nan_prob_no_reason": 0.0,
            "max_num_classes": 10,
            "categorical_feature_p": 0.0,
            "semantic_feature_p": 0.3,
            "random_seed": 42
        }
        
        # Create mock PriorBag
        self.prior_bag = MagicMock()
        
        # Setup default adapter
        self.adapter = ClassificationAdapter(self.prior_bag, self.prior_config)
        
        # Sample feature data
        n_samples = 10
        batch_size = 4
        self.num_features = 20
        self.x_data = torch.randn(n_samples, batch_size, self.num_features, device=self.device)
        
        # Define semantic feature indices
        self.semantic_features = list(range(self.num_features - 6, self.num_features))
    
    def tearDown(self):
        """Clean up temporary files"""
        self.temp_dir.cleanup()
        
    def test_load_semantic_prior_data(self):
        """Test that semantic prior data can be loaded correctly"""
        start_time = time.time()
        
        # Load from our test file
        column_tokens, value_tokens = load_semantic_prior_data(
            log_file_path=self.temp_json_path,
            force_reload=True,
            use_cache=False
        )
        
        load_time = time.time() - start_time
        print(f"load_semantic_prior_data took {load_time:.4f} seconds")
        
        # Verify returned data
        self.assertIsInstance(column_tokens, dict)
        self.assertIsInstance(value_tokens, dict)
        self.assertEqual(len(column_tokens), 10)  # 10 test columns
        
        # Check that all columns are tokenized
        for i in range(10):
            column_name = f"test_column_{i}"
            self.assertIn(column_name, column_tokens)
    
    def test_get_random_semantic_data(self):
        """Test random semantic data generation"""
        start_time = time.time()
        
        # Get random semantic data
        semantic_data, semantic_names = get_random_semantic_data(
            num_classes=5,
            use_cache=False
        )
        
        random_data_time = time.time() - start_time
        print(f"get_random_semantic_data took {random_data_time:.4f} seconds")
        
        # Verify data
        self.assertIsInstance(semantic_data, torch.Tensor)
        self.assertEqual(semantic_data.shape[0], 5)  # 5 classes
        self.assertGreater(semantic_data.shape[1], 10)  # Multiple tokens per class
    
    @patch('ticl.priors.classification_adapter.semantic_data')
    @patch('ticl.priors.classification_adapter.semantic_data_column_names')
    def test_apply_semantic_prior(self, mock_column_names, mock_semantic_data):
        """Test applying semantic features to data"""
        # Set up mock semantic data
        n_classes = 5
        n_semantic_features = len(self.semantic_features)
        
        # Create mock global semantic data
        mock_data = torch.randint(1000, 2000, (n_classes, 50))
        mock_semantic_data.__len__ = lambda: len(mock_data)
        mock_semantic_data.shape = mock_data.shape
        mock_semantic_data.to.return_value = mock_data
        mock_column_names.__len__ = lambda: n_classes
        
        # Apply semantic prior function
        start_time = time.time()
        
        x_new, semantic_info = self.adapter._apply_semantic_prior(
            self.x_data, 
            self.semantic_features, 
            self.device
        )
        
        apply_time = time.time() - start_time
        print(f"_apply_semantic_prior took {apply_time:.4f} seconds")
        
        # Verify results
        self.assertEqual(x_new.shape, self.x_data.shape)
        self.assertIn('semantic_targets', semantic_info)
        
        # Check that semantic features were actually modified
        self.assertFalse(torch.allclose(
            x_new[:, :, self.semantic_features], 
            self.x_data[:, :, self.semantic_features]
        ))
        
    def test_end_to_end_performance(self):
        """Performance test for the entire pipeline"""
        # Setup ClassificationAdapter with real semantic options
        self.prior_config["semantic_feature_p"] = 0.3
        adapter = ClassificationAdapter(self.prior_bag, self.prior_config)
        
        # Make sure semantic data is not cached from previous runs
        if hasattr(adapter, '_semantic_data_cache'):
            delattr(adapter, '_semantic_data_cache')
        
        # Test multiple iterations (optional - comment out if too slow)
        iterations = 3
        total_time = 0
        
        for i in range(iterations):
            # Create fresh data to avoid caching effects
            x_data = torch.randn(10, 4, self.num_features, device=self.device)
            
            # Apply semantic prior, measuring time
            start_time = time.time()
            
            with patch('ticl.datasets.semantic_prior_data_loader.load_semantic_prior_data',
                      return_value=(MagicMock(), MagicMock())):
                # Create mock for get_random_semantic_data
                mock_semantic_data = torch.randint(1000, 2000, (5, 50))
                mock_names = [f"feature_{i}" for i in range(5)]
                
                with patch('ticl.datasets.semantic_prior_data_loader.get_random_semantic_data',
                          return_value=(mock_semantic_data, mock_names)):
                    try:
                        x_new, semantic_info = adapter._apply_semantic_prior(
                            x_data, 
                            self.semantic_features, 
                            self.device
                        )
                    except Exception as e:
                        self.fail(f"_apply_semantic_prior raised {type(e).__name__}: {e}")
            
            iter_time = time.time() - start_time
            total_time += iter_time
            print(f"Iteration {i+1}: {iter_time:.4f} seconds")
        
        avg_time = total_time / iterations
        print(f"Average time over {iterations} iterations: {avg_time:.4f} seconds")


if __name__ == '__main__':
    unittest.main()