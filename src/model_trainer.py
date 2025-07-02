"""
Model Training Module
Handles model training, hyperparameter tuning, and evaluation with MLflow tracking.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A comprehensive model training class with MLflow integration.
    """
    
    def __init__(self, experiment_name="credit_risk_modeling", random_state=42):
        """
        Initialize the ModelTrainer.
        
        Args:
            experiment_name (str): MLflow experiment name
            random_state (int): Random state for reproducibility
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the models to be trained."""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state)
        }
    
    def split_data(self, X, y, test_size=0.2, stratify=None):
        """
        Split the data into training and testing sets.
        
        Args:
            X: Features
            y: Target variable
            test_size (float): Proportion of dataset to include in test split
            stratify: If not None, data is split in a stratified fashion
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if stratify is None:
            stratify = y
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )
        
        # Store splits for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training target distribution: {y_train.value_counts().to_dict()}")
        print(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def get_hyperparameter_grids(self):
        """
        Get hyperparameter grids for each model.
        
        Returns:
            dict: Parameter grids for each model
        """
        return {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    
    def train_model(self, model_name, X_train=None, y_train=None, 
                   hyperparameter_tuning=True, tuning_method='grid', cv=5):
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model to train
            X_train: Training features (optional, uses stored if None)
            y_train: Training target (optional, uses stored if None)
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            tuning_method (str): 'grid' or 'random' search
            cv (int): Number of cross-validation folds
            
        Returns:
            Trained model
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("hyperparameter_tuning", hyperparameter_tuning)
            mlflow.log_param("tuning_method", tuning_method)
            mlflow.log_param("cv_folds", cv)
            mlflow.log_param("training_samples", len(X_train))
            
            model = self.models[model_name]
            
            if hyperparameter_tuning:
                param_grid = self.get_hyperparameter_grids()[model_name]
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        model, param_grid, cv=cv, scoring='roc_auc', 
                        n_jobs=-1, verbose=1
                    )
                else:  # random search
                    search = RandomizedSearchCV(
                        model, param_grid, cv=cv, scoring='roc_auc',
                        n_jobs=-1, verbose=1, n_iter=20, random_state=self.random_state
                    )
                
                print(f"Starting {tuning_method} search for {model_name}...")
                search.fit(X_train, y_train)
                
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_cv_score = search.best_score_
                
                # Log hyperparameters
                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)
                
                mlflow.log_metric("best_cv_roc_auc", best_cv_score)
                
                print(f"Best parameters for {model_name}: {best_params}")
                print(f"Best CV ROC-AUC: {best_cv_score:.4f}")
                
            else:
                print(f"Training {model_name} with default parameters...")
                best_model = model
                best_model.fit(X_train, y_train)
            
            # Store the trained model
            self.trained_models[model_name] = best_model
            
            # Log the model
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(best_model, f"{model_name}_model", signature=signature)
            
            return best_model
    
    def train_all_models(self, X_train=None, y_train=None, hyperparameter_tuning=True):
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
            
        print("Training all models...")
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*50}")
            self.train_model(model_name, X_train, y_train, hyperparameter_tuning)
    
    def evaluate_model(self, model_name, X_test=None, y_test=None):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
        
        model = self.trained_models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Log metrics with MLflow
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        return metrics
    
    def evaluate_all_models(self, X_test=None, y_test=None):
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            pd.DataFrame: Comparison of all model metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        print("Evaluating all models...")
        evaluation_results = {}
        
        for model_name in self.trained_models.keys():
            print(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(model_name, X_test, y_test)
            evaluation_results[model_name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(evaluation_results).T
        comparison_df = comparison_df.round(4)
        
        # Find best model based on ROC-AUC (or F1 if ROC-AUC not available)
        if 'roc_auc' in comparison_df.columns:
            best_model_name = comparison_df['roc_auc'].idxmax()
            self.best_score = comparison_df.loc[best_model_name, 'roc_auc']
        else:
            best_model_name = comparison_df['f1_score'].idxmax()
            self.best_score = comparison_df.loc[best_model_name, 'f1_score']
        
        self.best_model = self.trained_models[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best score: {self.best_score:.4f}")
        
        return comparison_df
    
    def plot_model_comparison(self, save_path=None):
        """
        Plot comparison of model performance.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.results:
            print("No evaluation results found. Please evaluate models first.")
            return
        
        # Extract metrics for plotting
        models = list(self.results.keys())
        metrics_data = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']}
        
        for model_name in models:
            for metric in metrics_data.keys():
                if metric in self.results[model_name]['metrics']:
                    metrics_data[metric].append(self.results[model_name]['metrics'][metric])
                else:
                    metrics_data[metric].append(0)
        
        # Create subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot each metric
        for idx, (metric, values) in enumerate(metrics_data.items()):
            row = idx // 3
            col = idx % 3
            
            if idx < 5:  # We have 5 metrics
                ax = axes[row, col]
                bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
                ax.set_title(f'{metric.upper()}', fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)
        
        # Remove the last subplot (we only have 5 metrics)
        axes[1, 2].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def register_best_model(self, model_name=None):
        """
        Register the best model in MLflow Model Registry.
        
        Args:
            model_name (str): Name for the registered model
        """
        if self.best_model is None:
            print("No best model found. Please evaluate models first.")
            return
        
        if model_name is None:
            model_name = "credit_risk_best_model"
        
        # Find the best model name
        best_model_name = None
        for name, model in self.trained_models.items():
            if model == self.best_model:
                best_model_name = name
                break
        
        with mlflow.start_run(run_name=f"register_{model_name}"):
            # Log the best model
            signature = infer_signature(self.X_train, self.best_model.predict(self.X_train))
            model_uri = mlflow.sklearn.log_model(
                self.best_model, 
                "model", 
                signature=signature,
                registered_model_name=model_name
            ).model_uri
            
            # Log best model info
            mlflow.log_param("best_model_type", best_model_name)
            mlflow.log_metric("best_model_score", self.best_score)
            
            print(f"Best model ({best_model_name}) registered as '{model_name}' in MLflow Model Registry")
            print(f"Model URI: {model_uri}")
            
            return model_uri
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet.")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute.")
            return None

print("Model Trainer module loaded successfully!") 