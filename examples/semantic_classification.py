"""
Example demonstrating the use of semantic features and text-based classification.

This script shows how to:
1. Create a model with semantic features
2. Train it on a simple dataset
3. Use text descriptions for classification
4. Generate new class boundaries from text

Usage:
    python -m examples.semantic_classification
"""

import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ticl.model_builder import get_model
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data
from ticl.text_classifier import TextualClassifier


def train_example():
    """Train a simple model with semantic features for demonstration."""
    print("This would train a model with semantic features enabled.")
    print("For example:")
    print("python -m ticl.fit_model tabpfn --semantic-feature-p 0.3 --epochs 100")
    print("\nFor this example, we'll load a pre-configured model.")


def text_classification_example():
    """Demonstrate text-based classification using a model with semantic features."""
    # Load Iris dataset for demonstration
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create model configuration
    config = {
        'model_type': 'tabpfn',
        'semantic_prediction': True,
        'num_semantic_classes': 3,
        'prior': {
            'classification': {
                'semantic_feature_p': 0.3,
                'max_num_classes': 3
            },
            'num_features': 4
        }
    }
    
    # In a real scenario, we would load a trained model
    # For this example, we'll create a new model without training
    print("\nCreating a semantic-aware model...")
    _, model, _, _ = get_model(config, device='cpu', should_train=False)
    
    print("Creating a TextualClassifier...")
    semantic_data = get_random_semantic_data()
    classifier = TextualClassifier(model, semantic_data)
    
    # Convert test data to torch tensor with appropriate shape for demonstration
    # Real data would have been properly preprocessed
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Example 1: Classify using a single text description
    print("\n1. Classifying using a text description...")
    iris_description = "Flowers with large petal width and length"
    predictions, similarity = classifier.classify_with_text(
        X_test_tensor, 
        iris_description
    )
    print(f"Text description: '{iris_description}'")
    print(f"Similarity score: {similarity:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Example 2: Generate new class boundaries from text descriptions
    print("\n2. Generating new class boundaries from text descriptions...")
    class_descriptions = {
        "setosa": "Flowers with small petals and small sepals",
        "versicolor": "Flowers with medium-sized petals and sepals",
        "virginica": "Flowers with large petals and sepals"
    }
    
    predictions, class_mapping = classifier.classify_with_descriptions(
        X_test_tensor,
        class_descriptions
    )
    
    print("Class descriptions:")
    for name, desc in class_descriptions.items():
        print(f"  - {name}: {desc}")
    
    print(f"Class mapping: {class_mapping}")
    print(f"Predictions shape: {predictions.shape}")
    
    # For real evaluation, we'd compare to ground truth
    accuracy = (predictions == torch.tensor(y_test).unsqueeze(1)).float().mean()
    print(f"Note: This is a demonstration with untrained models, so accuracy is random: {accuracy:.4f}")


if __name__ == "__main__":
    print("=== Semantic Features and Text-Based Classification Example ===\n")
    
    train_example()
    text_classification_example()
    
    print("\n=== Example Complete ===")
    print("For real applications, you would:")
    print("1. Train a model with semantic features enabled")
    print("2. Save the model using the provided checkpointing mechanism")
    print("3. Load the trained model for text-based classification")
    print("4. Use the TextualClassifier with actual text descriptions relevant to your domain")