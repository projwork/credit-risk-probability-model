"""
Model Evaluation Module
Provides comprehensive model evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation class with visualization capabilities.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.evaluation_results = {}
    
    def evaluate_binary_classification(self, y_true, y_pred, y_pred_proba=None, 
                                     model_name="Model", pos_label=1):
        """
        Comprehensive evaluation for binary classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name (str): Name of the model
            pos_label: Positive class label
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, matthews_corrcoef,
            balanced_accuracy_score, cohen_kappa_score
        )
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=pos_label),
            'recall': recall_score(y_true, y_pred, pos_label=pos_label),
            'f1_score': f1_score(y_true, y_pred, pos_label=pos_label),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy and other metrics as text
        metrics = self.evaluation_results[model_name]['metrics']
        info_text = f"Accuracy: {metrics['accuracy']:.3f}\n"
        info_text += f"Precision: {metrics['precision']:.3f}\n"
        info_text += f"Recall: {metrics['recall']:.3f}\n"
        info_text += f"F1-Score: {metrics['f1_score']:.3f}"
        
        plt.text(1.05, 0.5, info_text, transform=plt.gca().transAxes,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, model_names=None, figsize=(10, 8), save_path=None):
        """
        Plot ROC curve for one or multiple models.
        
        Args:
            model_names (list): List of model names to plot (None for all)
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=figsize)
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"No evaluation results found for {model_name}")
                continue
            
            results = self.evaluation_results[model_name]
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            if y_pred_proba is None:
                print(f"No prediction probabilities found for {model_name}")
                continue
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, model_names=None, figsize=(10, 8), save_path=None):
        """
        Plot Precision-Recall curve for one or multiple models.
        
        Args:
            model_names (list): List of model names to plot (None for all)
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=figsize)
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"No evaluation results found for {model_name}")
                continue
            
            results = self.evaluation_results[model_name]
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            if y_pred_proba is None:
                print(f"No prediction probabilities found for {model_name}")
                continue
            
            # Calculate Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            # Plot
            plt.plot(recall, precision, linewidth=2,
                    label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison', fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, model_names=None, n_bins=10, figsize=(10, 8), save_path=None):
        """
        Plot calibration curve for probability predictions.
        
        Args:
            model_names (list): List of model names to plot (None for all)
            n_bins (int): Number of bins for calibration
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        plt.figure(figsize=figsize)
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                print(f"No evaluation results found for {model_name}")
                continue
            
            results = self.evaluation_results[model_name]
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            if y_pred_proba is None:
                print(f"No prediction probabilities found for {model_name}")
                continue
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            # Plot
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    linewidth=2, label=f'{model_name}')
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, top_n=20, 
                              figsize=(12, 8), save_path=None):
        """
        Plot feature importance.
        
        Args:
            feature_importance_df (pd.DataFrame): DataFrame with features and importance
            top_n (int): Number of top features to plot
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if feature_importance_df is None or feature_importance_df.empty:
            print("No feature importance data provided")
            return
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances', fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def create_evaluation_report(self, model_names=None):
        """
        Create a comprehensive evaluation report.
        
        Args:
            model_names (list): List of model names to include (None for all)
            
        Returns:
            pd.DataFrame: Summary report
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        report_data = []
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                continue
            
            metrics = self.evaluation_results[model_name]['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.round(4)
        
        # Sort by ROC-AUC if available, otherwise by F1-score
        if 'roc_auc' in report_df.columns:
            report_df = report_df.sort_values('roc_auc', ascending=False)
        elif 'f1_score' in report_df.columns:
            report_df = report_df.sort_values('f1_score', ascending=False)
        
        return report_df
    
    def plot_evaluation_dashboard(self, model_name, figsize=(20, 15), save_path=None):
        """
        Create a comprehensive evaluation dashboard for a single model.
        
        Args:
            model_name (str): Name of the model
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Model Evaluation Dashboard - {model_name}', fontsize=16, fontweight='bold')
        
        results = self.evaluation_results[model_name]
        y_true = results['y_true']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        cm = results['confusion_matrix']
        metrics = results['metrics']
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            axes[0, 2].plot(recall, precision, linewidth=2, label=f'AP = {avg_precision:.3f}')
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curve')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Metrics Bar Plot
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        bars = axes[1, 0].bar(metric_names, metric_values, color='skyblue')
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Prediction Distribution
        if y_pred_proba is not None:
            axes[1, 1].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='Class 0', color='red')
            axes[1, 1].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='Class 1', color='blue')
            axes[1, 1].set_xlabel('Predicted Probability')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Prediction Probability Distribution')
            axes[1, 1].legend()
        
        # 6. Calibration Curve
        if y_pred_proba is not None:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            axes[1, 2].plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2)
            axes[1, 2].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axes[1, 2].set_xlabel('Mean Predicted Probability')
            axes[1, 2].set_ylabel('Fraction of Positives')
            axes[1, 2].set_title('Calibration Curve')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation dashboard saved to {save_path}")
        
        plt.show()

print("Model Evaluator module loaded successfully!") 