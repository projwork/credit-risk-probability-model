"""
Data Loading Module
Handles loading and basic preprocessing of the credit risk dataset.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Data loading and basic preprocessing class."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path (str): Path to the data directory
        """
        if data_path is None:
            # Default to relative path from src directory
            self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        else:
            self.data_path = data_path
        
        self.raw_data_path = os.path.join(self.data_path, 'raw')
        self.processed_data_path = os.path.join(self.data_path, 'processed')
        
        # Ensure directories exist
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        self.data = None
        self.feature_columns = None
        self.target_column = 'FraudResult'
    
    def load_raw_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            filename (str): Name of the CSV file to load
            
        Returns:
            pd.DataFrame: Loaded raw data
        """
        if filename is None:
            # Try to find CSV files in raw data directory
            csv_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the raw data directory")
            filename = csv_files[0]  # Use the first CSV file found
            print(f"Using data file: {filename}")
        
        file_path = os.path.join(self.raw_data_path, filename)
        
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {filename}")
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            # Set feature columns (excluding target)
            if self.target_column in self.data.columns:
                self.feature_columns = [col for col in self.data.columns if col != self.target_column]
            else:
                self.feature_columns = list(self.data.columns)
                print(f"Warning: Target column '{self.target_column}' not found in data")
            
            return self.data
        
        except Exception as e:
            raise Exception(f"Error loading data from {filename}: {str(e)}")
    
    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            dict: Data information summary
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_raw_data()")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.data.duplicated().sum()
        }
        
        # Target variable information (if available)
        if self.target_column in self.data.columns:
            target_info = {
                'target_distribution': self.data[self.target_column].value_counts().to_dict(),
                'target_percentages': (self.data[self.target_column].value_counts(normalize=True) * 100).to_dict()
            }
            info.update(target_info)
        
        # Numeric columns statistics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            info['numeric_columns'] = numeric_cols
            info['numeric_stats'] = self.data[numeric_cols].describe().to_dict()
        
        # Categorical columns information
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            info['categorical_columns'] = categorical_cols
            info['categorical_unique_counts'] = {col: self.data[col].nunique() for col in categorical_cols}
        
        return info
    
    def prepare_model_data(self, target_column: str = None, 
                          exclude_columns: list = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            target_column (str): Name of the target column
            exclude_columns (list): Columns to exclude from features
            
        Returns:
            tuple: (X, y) - Features and target
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_raw_data()")
        
        if target_column is None:
            target_column = self.target_column
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Prepare feature columns
        feature_cols = [col for col in self.data.columns if col != target_column]
        
        # Exclude specified columns
        if exclude_columns:
            feature_cols = [col for col in feature_cols if col not in exclude_columns]
        
        # Extract features and target
        X = self.data[feature_cols].copy()
        y = self.data[target_column].copy()
        
        print(f"Prepared data for modeling:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_proxy_target(self, method: str = 'fraud_based') -> pd.Series:
        """
        Create proxy target variable when direct labels are not available.
        
        Args:
            method (str): Method to create proxy target
            
        Returns:
            pd.Series: Proxy target variable
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_raw_data()")
        
        if method == 'fraud_based' and 'FraudResult' in self.data.columns:
            # Use existing fraud labels as proxy for credit risk
            proxy_target = self.data['FraudResult'].copy()
            print("Using FraudResult as proxy for credit risk")
            
        elif method == 'amount_based':
            # Create proxy based on transaction amount (high amounts = higher risk)
            if 'Amount' in self.data.columns:
                amount_threshold = self.data['Amount'].quantile(0.95)
                proxy_target = (self.data['Amount'] > amount_threshold).astype(int)
                print(f"Created amount-based proxy target (threshold: {amount_threshold})")
            else:
                raise ValueError("Amount column not found for amount-based proxy")
                
        elif method == 'frequency_based':
            # Create proxy based on customer transaction frequency
            if 'CustomerId' in self.data.columns:
                customer_freq = self.data['CustomerId'].value_counts()
                high_freq_customers = customer_freq[customer_freq > customer_freq.quantile(0.9)].index
                proxy_target = self.data['CustomerId'].isin(high_freq_customers).astype(int)
                print("Created frequency-based proxy target")
            else:
                raise ValueError("CustomerId column not found for frequency-based proxy")
        
        else:
            raise ValueError(f"Unknown proxy method: {method}")
        
        return proxy_target
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """
        Save processed data to CSV file.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Name of the output file
        """
        file_path = os.path.join(self.processed_data_path, filename)
        data.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    
    def sample_data(self, n_samples: int = None, fraction: float = None, 
                   random_state: int = 42) -> pd.DataFrame:
        """
        Sample data for faster processing during development.
        
        Args:
            n_samples (int): Number of samples to take
            fraction (float): Fraction of data to sample
            random_state (int): Random state for reproducibility
            
        Returns:
            pd.DataFrame: Sampled data
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_raw_data()")
        
        if n_samples is not None:
            if n_samples > len(self.data):
                print(f"Warning: Requested {n_samples} samples, but only {len(self.data)} available")
                return self.data.copy()
            sampled_data = self.data.sample(n=n_samples, random_state=random_state)
        elif fraction is not None:
            sampled_data = self.data.sample(frac=fraction, random_state=random_state)
        else:
            raise ValueError("Either n_samples or fraction must be specified")
        
        print(f"Sampled {len(sampled_data)} rows from {len(self.data)} total rows")
        return sampled_data

def load_credit_risk_data(data_path: str = None, sample_size: int = None) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to load credit risk data.
    
    Args:
        data_path (str): Path to data directory
        sample_size (int): Number of samples to load (None for all data)
        
    Returns:
        tuple: (data, data_info)
    """
    loader = DataLoader(data_path)
    data = loader.load_raw_data()
    
    if sample_size is not None:
        data = loader.sample_data(n_samples=sample_size)
        loader.data = data  # Update loader's data reference
    
    data_info = loader.get_data_info()
    
    return data, data_info

print("Data Loader module loaded successfully!") 