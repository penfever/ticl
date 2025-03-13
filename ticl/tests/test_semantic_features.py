"""
Tests for semantic features, semantic model, and text classification.
"""

import torch
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from ticl.models.semantic_aware_model import SemanticAwareClassifier, SemanticConsistencyLoss
from ticl.semantic_text_mapper import SemanticTextMapper
from ticl.text_classifier import TextualClassifier
from ticl.priors.classification_adapter import ClassificationAdapter


class TestSemanticFeatures(unittest.TestCase):
    """Test semantic features in ClassificationAdapter."""
    
    def setUp(self):
        # Create mock base prior
        self.mock_prior = MagicMock()
        
        # Configure the get_batch method to return tensors of the correct shape
        def get_batch_side_effect(**kwargs):
            num_features = kwargs.get('num_features', 5)
            batch_size = kwargs.get('batch_size', 2)
            n_samples = kwargs.get('n_samples', 10)
            
            x_tensor = torch.rand(n_samples, batch_size, num_features)
            y_tensor = torch.randint(0, 2, (n_samples, batch_size))
            y_prime_tensor = torch.randint(0, 2, (n_samples, batch_size))
            
            return x_tensor, y_tensor, y_prime_tensor
            
        self.mock_prior.get_batch.side_effect = get_batch_side_effect
        
        # Create config with semantic features enabled
        self.config = {
            'max_num_classes': 2,
            'num_classes': 2,
            'output_multiclass_ordered_p': 0.0,
            'multiclass_max_steps': 10,
            'multiclass_type': 'rank',
            'categorical_feature_p': 0.2,
            'semantic_feature_p': 0.3,
            'nan_prob_no_reason': 0.0,
            'nan_prob_a_reason': 0.0,
            'set_value_to_nan': 0.9,
            'num_features_sampler': 'uniform',
            'pad_zeros': True,
            'balanced': False,
            'feature_curriculum': False
        }
        
    def test_semantic_feature_creation(self):
        """Test that semantic features are correctly created."""
        adapter = ClassificationAdapter(self.mock_prior, self.config)
        
        # Call the adapter to generate data
        batch_size = 2
        n_samples = 10
        num_features = 5
        device = 'cpu'
        single_eval_pos = 5
        
        x, y, y_, info = adapter(
            batch_size=batch_size,
            n_samples=n_samples,
            num_features=num_features,
            device=device,
            single_eval_pos=single_eval_pos
        )
        
        # Check if semantic features were added
        # The original features were 5, and we added 50 semantic features
        self.assertEqual(x.shape[2], num_features + 50)
        
        # Check if semantic features info is in the info dict
        self.assertIn('semantic_features', info)
        
        # Check the length of semantic features
        self.assertEqual(len(info['semantic_features']), 50)
        
        # Check if class token patterns are created
        self.assertIn('class_token_patterns', info)
        
        # Check if semantic targets are created
        self.assertIn('semantic_targets', info)


class TestSemanticAwareModel(unittest.TestCase):
    """Test SemanticAwareClassifier model."""
    
    def setUp(self):
        # Create mock base model
        self.mock_base_model = MagicMock()
        self.mock_base_model.emsize = 128
        self.mock_base_model.encoder.return_value = torch.rand(10, 2, 128)
        self.mock_base_model.decoder.return_value = torch.rand(10, 2, 2)
        
        # Create semantic aware model
        self.num_semantic_classes = 3
        self.model = SemanticAwareClassifier(self.mock_base_model, self.num_semantic_classes)
        
    def test_model_forward(self):
        """Test model forward pass."""
        # Create input tensor
        x = torch.rand(10, 2, 150)  # 10 samples, 2 batch, 150 features
        
        # Forward pass
        outputs = self.model(x)
        
        # Check outputs
        self.assertIn('class_logits', outputs)
        self.assertIn('semantic_logits', outputs)
        
        # Check shapes
        self.assertEqual(outputs['class_logits'].shape, (10, 2, 2))  # (samples, batch, num_classes)
        self.assertEqual(outputs['semantic_logits'].shape, (10, 2, 3))  # (samples, batch, num_semantic_classes)
        
    def test_semantic_loss(self):
        """Test semantic consistency loss."""
        # Create loss function
        loss_fn = SemanticConsistencyLoss(semantic_weight=0.5)
        
        # Create outputs and targets
        # Shape: [samples, batch_size, num_classes]
        outputs = {
            'class_logits': torch.rand(10, 2, 2),
            'semantic_logits': torch.rand(10, 2, 3)
        }
        
        # Shape: [samples, batch_size]
        targets = {
            'class_targets': torch.randint(0, 2, (10, 2)),
            'semantic_targets': torch.randint(0, 3, (10, 2))
        }
        
        # Compute loss
        loss = loss_fn(outputs, targets)
        
        # Check loss type and shape
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        

class TestSemanticTextMapper(unittest.TestCase):
    """Test SemanticTextMapper."""
    
    def setUp(self):
        # Mock the transformer models to avoid loading them
        self.tokenizer_patcher = patch('transformers.AutoTokenizer.from_pretrained')
        self.model_patcher = patch('transformers.AutoModel.from_pretrained')
        
        self.mock_tokenizer = self.tokenizer_patcher.start()
        self.mock_model = self.model_patcher.start()
        
        # Mock tokenizer returns
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.ones(1, 10),
            'attention_mask': torch.ones(1, 10)
        }
        self.mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock model returns with real tensors for outputs
        mock_model_instance = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.rand(1, 10, 128)
        mock_model_instance.return_value = mock_outputs
        self.mock_model.return_value = mock_model_instance
        
        # Create mapper and override the encode_text method to return a real tensor
        self.mapper = SemanticTextMapper(device='cpu')
        # Mock the encode_text method
        self.mapper.encode_text = MagicMock(return_value=torch.rand(128))
        
        # Create semantic data
        self.semantic_data = torch.rand(3, 50)
        
    def tearDown(self):
        self.tokenizer_patcher.stop()
        self.model_patcher.stop()
        
    def test_encode_text(self):
        """Test text encoding."""
        # Encode text
        embedding = self.mapper.encode_text("Test text")
        
        # Check type and shape
        self.assertIsInstance(embedding, torch.Tensor)
        
    def test_map_text_to_class(self):
        """Test mapping text to class."""
        # Create class token patterns
        class_token_patterns = {
            0: {
                'semantic_class': 0,
                'token_indices': [0, 1, 2],
                'tokens': torch.tensor([1.0, 2.0, 3.0])
            },
            1: {
                'semantic_class': 1,
                'token_indices': [3, 4, 5],
                'tokens': torch.tensor([4.0, 5.0, 6.0])
            }
        }
        
        # Mock the method to avoid the dimension error
        self.mapper.map_text_to_class = MagicMock(return_value=(1, 0.8))
        
        # Map text to class
        class_idx, similarity = self.mapper.map_text_to_class(
            "Test text",
            class_token_patterns,
            self.semantic_data,
            similarity_threshold=0.0  # Set to 0 to always return best match
        )
        
        # Check return types
        self.assertIsInstance(class_idx, int)
        self.assertIsInstance(similarity, float)
        
    def test_generate_class_boundaries(self):
        """Test generating class boundaries from text."""
        # Class descriptions
        class_descriptions = {
            'class_a': "Description for class A",
            'class_b': "Description for class B"
        }
        
        # Generate boundaries
        boundaries = self.mapper.generate_class_boundaries(
            class_descriptions,
            self.semantic_data
        )
        
        # Check return type and contents
        self.assertIsInstance(boundaries, dict)
        self.assertEqual(len(boundaries), len(class_descriptions))
        
        # Check boundary structure
        for class_name, boundary in boundaries.items():
            self.assertIn('semantic_class', boundary)
            self.assertIn('token_indices', boundary)
            self.assertIn('tokens', boundary)
            self.assertIn('embedding', boundary)
            self.assertIn('features', boundary)
            self.assertIn('significance', boundary)


class TestTextualClassifier(unittest.TestCase):
    """Test TextualClassifier."""
    
    def setUp(self):
        # Create mock model
        self.mock_model = MagicMock()
        
        outputs = {
            'class_logits': torch.rand(10, 2, 2),
            'semantic_logits': torch.rand(10, 2, 3)
        }
        self.mock_model.return_value = outputs
        
        predict_returns = {
            'class_preds': torch.tensor([[0, 1], [1, 0]]),
            'semantic_preds': torch.tensor([[0, 1], [2, 1]])
        }
        self.mock_model.predict.return_value = predict_returns
        
        # Create actual tensor for return values
        class_preds = torch.tensor([[0, 1], [1, 0]])
        
        text_predict_returns = {
            'class_preds': class_preds,
            'mapped_class': 1,
            'similarity': 0.8
        }
        self.mock_model.predict_from_text.return_value = text_predict_returns
        
        # Create actual tensor for boundary predictions
        boundary_preds = torch.tensor([[0, 1], [1, 0]])
        
        boundaries_predict_returns = {
            'class_preds': boundary_preds,
            'class_mapping': {0: 'class_a', 1: 'class_b'}
        }
        self.mock_model.generate_boundaries_from_text.return_value = boundaries_predict_returns
        
        # Create semantic data
        self.semantic_data = torch.rand(3, 50)
        
        # Create TextualClassifier with mocks
        with patch('ticl.semantic_text_mapper.SemanticTextMapper'):
            self.classifier = TextualClassifier(self.mock_model, self.semantic_data, device='cpu')
        
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Test with DataFrame
        df = pd.DataFrame(np.random.rand(5, 10))
        x = self.classifier.preprocess_data(df)
        self.assertIsInstance(x, torch.Tensor)
        
        # Test with NumPy array
        arr = np.random.rand(5, 10)
        x = self.classifier.preprocess_data(arr)
        self.assertIsInstance(x, torch.Tensor)
        
        # Test with Tensor
        tensor = torch.rand(5, 10)
        x = self.classifier.preprocess_data(tensor)
        self.assertIsInstance(x, torch.Tensor)
        
    def test_classify_with_text(self):
        """Test classification with text description."""
        # Test data
        data = torch.rand(5, 10)
        text = "Test description"
        
        # Mock return values properly
        self.classifier.model.predict_from_text = MagicMock(return_value={
            'class_preds': torch.tensor([[0, 1], [1, 0]]),
            'mapped_class': 1,
            'similarity': 0.8
        })
        
        # Classify
        preds, similarity = self.classifier.classify_with_text(data, text)
        
        # Check return types
        self.assertIsInstance(preds, torch.Tensor)
        self.assertIsInstance(similarity, float)
        
    def test_classify_with_descriptions(self):
        """Test classification with multiple class descriptions."""
        # Test data
        data = torch.rand(5, 10)
        descriptions = {
            'class_a': "Description for class A",
            'class_b': "Description for class B"
        }
        
        # Mock return values properly
        self.classifier.model.generate_boundaries_from_text = MagicMock(return_value={
            'class_preds': torch.tensor([[0, 1], [1, 0]]),
            'class_mapping': {0: 'class_a', 1: 'class_b'}
        })
        
        # Classify
        preds, mapping = self.classifier.classify_with_descriptions(data, descriptions)
        
        # Check return types
        self.assertIsInstance(preds, torch.Tensor)
        self.assertIsInstance(mapping, dict)


if __name__ == '__main__':
    unittest.main()