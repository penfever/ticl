import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ticl.models.tabpfn import TabPFN
from ticl.models.semantic_aware_model import SemanticAwareClassifier, create_semantic_aware_model
from ticl.prediction.semantic import SemanticAwareClassifierWrapper


@pytest.fixture
def get_device():
    """Determine the best available device for testing."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


@pytest.fixture
def sample_config():
    """Create a sample configuration for a TabPFN model."""
    return {
        'model_type': 'tabpfn',
        'transformer': {
            'dim': 512,
            'depth': 2,
            'heads': 8,
            'mlp_ratio': 4,
            'dropout': 0.0,
            'classification_task': True,
            'semantic_feature_p': 0.3,
        },
        'prior': {
            'num_features': 100,
            'classification': {
                'max_num_classes': 10,
                'pad_zeros': True
            }
        },
        'openmlloader': {
            'max_samples': 1000,
            'pca': False,
            'valid_data': 'new'
        },
        'device': get_device,
    }


@pytest.fixture
def create_dummy_model(get_device, sample_config):
    """Create a dummy SemanticAwareClassifier model for testing."""
    device = get_device
    
    # Mock the transformers modules needed for CLIP
    clip_tokenizer_patcher = patch('transformers.CLIPTokenizerFast')
    clip_model_patcher = patch('transformers.CLIPTextModel')
    
    mock_tokenizer = clip_tokenizer_patcher.start()
    mock_text_model = clip_model_patcher.start()
    
    # Configure mock tokenizer
    tokenizer_instance = MagicMock()
    tokenizer_instance.return_value = {'input_ids': torch.ones(1, 77), 'attention_mask': torch.ones(1, 77)}
    mock_tokenizer.from_pretrained.return_value = tokenizer_instance
    
    # Configure mock text model
    text_model_instance = MagicMock()
    text_model_instance.config.hidden_size = 512
    pooler_output = torch.zeros(1, 512)
    outputs = MagicMock()
    outputs.pooler_output = pooler_output
    text_model_instance.return_value = outputs
    mock_text_model.from_pretrained.return_value = text_model_instance
    
    # Create a simple y encoder layer for TabPFN
    y_encoder = torch.nn.Sequential(
        torch.nn.Linear(1, 128),
        torch.nn.GELU()
    )
    
    # Create a small TabPFN as the base model
    base_model = TabPFN(
        emsize=128,               # Embedding size
        nhead=4,                  # Number of attention heads
        nhid_factor=4,            # Hidden dimension is emsize * nhid_factor
        nlayers=2,                # Number of transformer layers
        n_features=100,           # Number of input features
        n_out=10,                 # Number of output classes
        dropout=0.1,              # Dropout rate
        activation='gelu',        # Activation function
        y_encoder_layer=y_encoder,  # Required Y encoder layer
        semantic_feature_p=0.3    # Semantic feature probability
    )
    
    # Initialize weights with small random values
    for param in base_model.parameters():
        param.data.normal_(0, 0.02)
    
    # Move to the appropriate device
    base_model = base_model.to(device)
    
    try:
        # Create semantic-aware model with CLIP mocking
        with patch('transformers.CLIPTokenizerFast.from_pretrained', mock_tokenizer.from_pretrained), \
             patch('transformers.CLIPTextModel.from_pretrained', mock_text_model.from_pretrained):
            
            semantic_model = SemanticAwareClassifier(base_model, num_semantic_classes=3)
            semantic_model = semantic_model.to(device)
    finally:
        # Stop the patchers
        clip_tokenizer_patcher.stop()
        clip_model_patcher.stop()
    
    return semantic_model, sample_config


class TestSemanticAwareClassifierWrapper:
    """Test suite for SemanticAwareClassifierWrapper."""
    
    def test_wrapper_initialization(self, create_dummy_model, get_device):
        """Test if the wrapper can be properly initialized."""
        model, config = create_dummy_model
        device = get_device
        
        # Create wrapper
        wrapper = SemanticAwareClassifierWrapper(
            device=device,
            model=model,
            config=config,
            batch_size=8,
            verbose=True
        )
        
        # Check attributes
        assert wrapper.model is model
        assert wrapper.device == device
        assert wrapper.max_num_features == config['prior']['num_features']
        
    def test_wrapper_with_iris_dataset(self, create_dummy_model, get_device):
        """Test if the wrapper can fit and predict on the Iris dataset."""
        model, config = create_dummy_model
        device = get_device
        
        # Load Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Create class descriptions for semantic features
        class_descriptions = {
            'setosa': 'Iris setosa with short sepals and petals',
            'versicolor': 'Iris versicolor with medium sepals and petals',
            'virginica': 'Iris virginica with long sepals and petals'
        }
        
        # Create wrapper
        wrapper = SemanticAwareClassifierWrapper(
            device=device,
            model=model,
            config=config,
            batch_size=8,
            verbose=True,
            N_ensemble_configurations=2,  # Use smaller ensemble for faster testing
            semantic_class_descriptions=class_descriptions
        )
        
        # Fit the model
        wrapper.fit(X_train, y_train)
        
        # Make predictions
        y_pred = wrapper.predict(X_test)
        
        # Basic check: predictions have correct shape
        assert len(y_pred) == len(y_test)
        assert isinstance(y_pred, np.ndarray)
    
    def test_wrapper_with_breast_cancer_dataset(self, create_dummy_model, get_device):
        """Test if the wrapper can fit and predict on the Breast Cancer dataset."""
        model, config = create_dummy_model
        device = get_device
        
        # Load Breast Cancer dataset
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Create semantic columns - example of marking some columns as semantic
        semantic_column_indices = [0, 1, 2, 3, 4]  # First 5 columns as semantic
        
        # Create wrapper
        wrapper = SemanticAwareClassifierWrapper(
            device=device,
            model=model,
            config=config,
            batch_size=8,
            verbose=True,
            N_ensemble_configurations=2,  # Use smaller ensemble for faster testing
            semantic_column_indices=semantic_column_indices
        )
        
        # Fit the model
        wrapper.fit(X_train, y_train, semantic_column_indices=semantic_column_indices)
        
        # Get probabilities
        y_proba = wrapper.predict_proba(X_test)
        
        # Make predictions
        y_pred = wrapper.predict(X_test)
        
        # Check shapes and types
        assert y_proba.shape == (len(X_test), 2)
        assert len(y_pred) == len(y_test)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_proba, np.ndarray)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])