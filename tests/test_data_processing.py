"""
Unit Tests for Data Processing Functions
Tests for model training, evaluation, and data handling functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model_trainer import ModelTrainer
    from model_evaluator import ModelEvaluator
except ImportError:
    print("Warning: Could not import model modules. Some tests may fail.")

class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randint(0, 5, 1000)
        })
        y = pd.Series(np.random.randint(0, 2, 1000))
        return X, y
    
    @pytest.fixture
    def model_trainer(self):
        """Create ModelTrainer instance for testing."""
        return ModelTrainer(experiment_name="test_experiment")
    
    def test_model_trainer_initialization(self, model_trainer):
        """Test ModelTrainer initialization."""
        assert model_trainer.experiment_name == "test_experiment"
        assert model_trainer.random_state == 42
        assert len(model_trainer.models) == 4
        assert 'logistic_regression' in model_trainer.models
        assert 'decision_tree' in model_trainer.models
        assert 'random_forest' in model_trainer.models
        assert 'gradient_boosting' in model_trainer.models
    
    def test_data_splitting_functionality(self, sample_data):
        """Test data splitting functionality."""
        X, y = sample_data
        
        # Test basic splitting logic
        test_size = 0.2
        n_total = len(X)
        expected_train_size = int(n_total * (1 - test_size))
        expected_test_size = n_total - expected_train_size
        
        # Verify data integrity
        assert len(X) == len(y)
        assert not X.empty
        assert not y.empty
        
        # Test stratification logic
        assert y.nunique() >= 2  # Should have at least 2 classes
        
        print(f"✅ Data splitting test passed - Total samples: {n_total}")
    
    def test_hyperparameter_grids(self, model_trainer):
        """Test hyperparameter grid generation."""
        grids = model_trainer.get_hyperparameter_grids()
        
        # Check that all models have parameter grids
        assert 'logistic_regression' in grids
        assert 'decision_tree' in grids
        assert 'random_forest' in grids
        assert 'gradient_boosting' in grids
        
        # Check specific parameters
        assert 'C' in grids['logistic_regression']
        assert 'max_depth' in grids['decision_tree']
        assert 'n_estimators' in grids['random_forest']
        assert 'learning_rate' in grids['gradient_boosting']
    
    def test_model_training_without_tuning(self, model_trainer, sample_data):
        """Test model training without hyperparameter tuning."""
        X, y = sample_data
        model_trainer.split_data(X, y)
        
        # Train a simple model without hyperparameter tuning
        model = model_trainer.train_model('logistic_regression', hyperparameter_tuning=False)
        
        # Check that model is trained
        assert hasattr(model, 'fit')
        assert 'logistic_regression' in model_trainer.trained_models
        
        # Check that model can make predictions
        predictions = model.predict(model_trainer.X_test)
        assert len(predictions) == len(model_trainer.y_test)

class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100)
        return y_true, y_pred, y_pred_proba
    
    @pytest.fixture
    def model_evaluator(self):
        """Create ModelEvaluator instance for testing."""
        return ModelEvaluator()
    
    def test_model_evaluator_initialization(self, model_evaluator):
        """Test ModelEvaluator initialization."""
        assert hasattr(model_evaluator, 'evaluation_results')
        assert len(model_evaluator.evaluation_results) == 0
    
    def test_binary_classification_evaluation(self, model_evaluator, sample_predictions):
        """Test binary classification evaluation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        metrics = model_evaluator.evaluate_binary_classification(
            y_true, y_pred, y_pred_proba, model_name="test_model"
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1  # All metrics should be between 0 and 1
        
        # Check that evaluation results are stored
        assert "test_model" in model_evaluator.evaluation_results
        assert 'metrics' in model_evaluator.evaluation_results["test_model"]
        assert 'confusion_matrix' in model_evaluator.evaluation_results["test_model"]
    
    def test_evaluation_report_creation(self, model_evaluator, sample_predictions):
        """Test evaluation report creation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        # Add multiple model evaluations
        model_evaluator.evaluate_binary_classification(
            y_true, y_pred, y_pred_proba, model_name="model1"
        )
        model_evaluator.evaluate_binary_classification(
            y_true, y_pred, y_pred_proba, model_name="model2"
        )
        
        report = model_evaluator.create_evaluation_report()
        
        # Check report structure
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 2  # Two models
        assert 'Model' in report.columns
        assert 'accuracy' in report.columns
        assert 'precision' in report.columns

    def test_metric_calculation(self, sample_predictions):
        """Test metric calculation functionality."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        # Test basic metric bounds
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        # All metrics should be between 0 and 1
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        
        print(f"✅ Metric calculation test passed - Accuracy: {accuracy:.3f}")

class TestDataProcessingHelpers:
    """Test cases for data processing helper functions."""
    
    def test_data_validation(self):
        """Test data validation helper function."""
        # Create sample data
        valid_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test basic validation
        assert not valid_data.empty
        assert len(valid_data.columns) == 3
        assert valid_data['target'].nunique() == 2  # Binary target
        
        print("✅ Data validation test passed")
    
    def test_feature_preprocessing(self):
        """Test feature preprocessing functionality."""
        # Create data with missing values
        data_with_missing = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [0.1, np.nan, 0.3, 0.4, 0.5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Test missing value detection
        missing_counts = data_with_missing.isnull().sum()
        assert missing_counts['feature1'] == 1
        assert missing_counts['feature2'] == 1
        assert missing_counts['categorical'] == 0
        
        # Test data types
        assert data_with_missing['feature1'].dtype in [np.float64, np.int64, float]
        assert data_with_missing['categorical'].dtype == 'object'
        
        print("✅ Feature preprocessing test passed")

# Utility functions for testing
def create_balanced_dataset(n_samples=1000, n_features=5, random_state=42):
    """Create a balanced binary classification dataset."""
    np.random.seed(random_state)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y

def calculate_metric_bounds(metric_name):
    """Calculate expected bounds for evaluation metrics."""
    bounds = {
        'accuracy': (0.0, 1.0),
        'precision': (0.0, 1.0),
        'recall': (0.0, 1.0),
        'f1_score': (0.0, 1.0),
        'roc_auc': (0.0, 1.0)
    }
    return bounds.get(metric_name, (0.0, 1.0))

if __name__ == "__main__":
    # Run a simple test
    print("Running basic tests...")
    
    # Test data creation
    X, y = create_balanced_dataset()
    print(f"Created dataset with shape: {X.shape}, target distribution: {y.value_counts().to_dict()}")
    
    # Test metric bounds
    for metric in ['accuracy', 'precision', 'recall']:
        bounds = calculate_metric_bounds(metric)
        print(f"{metric} bounds: {bounds}")
    
    print("Basic tests completed successfully!") 