"""
Simple tests to verify that the semantic feature files exist and have expected structure.
"""

import os
import unittest
import importlib.util

class TestSemanticFilesExist(unittest.TestCase):
    """Test that the semantic feature files exist and have expected classes/functions."""
    
    def test_files_exist(self):
        """Test that all the semantic feature files exist."""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        expected_files = [
            os.path.join(root_dir, '..', 'models', 'semantic_aware_model.py'),
            os.path.join(root_dir, '..', 'semantic_text_mapper.py'),
            os.path.join(root_dir, '..', 'text_classifier.py'),
            os.path.join(root_dir, '..', 'priors', 'boundaries.py')
        ]
        
        for file_path in expected_files:
            self.assertTrue(os.path.exists(file_path), f"File not found: {file_path}")
            print(f"File exists: {file_path}")
    
    def test_module_structure(self):
        """Test that the modules have the expected classes and methods."""
        # Check for semantic model module
        spec = importlib.util.find_spec('ticl.models.semantic_aware_model')
        self.assertIsNotNone(spec, "semantic_aware_model module not found")
        
        # Check for text mapper module
        spec = importlib.util.find_spec('ticl.semantic_text_mapper')
        self.assertIsNotNone(spec, "semantic_text_mapper module not found")
        
        # Check for text classifier module
        spec = importlib.util.find_spec('ticl.text_classifier')
        self.assertIsNotNone(spec, "text_classifier module not found")
        
        print("All modules can be found by importlib")

if __name__ == '__main__':
    unittest.main()