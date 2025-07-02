"""
Temporal Features Module for Credit Risk Analysis
Extracts time-based features from transaction timestamps
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extracts temporal features from datetime columns"""
    
    def __init__(self, datetime_col='TransactionStartTime', 
                 extract_features=['hour', 'day', 'month', 'year', 'dayofweek']):
        self.datetime_col = datetime_col
        self.extract_features = extract_features
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        if self.datetime_col not in X_copy.columns:
            print(f"Warning: {self.datetime_col} not found in dataframe")
            return X_copy
            
        # Convert to datetime if not already
        X_copy[self.datetime_col] = pd.to_datetime(X_copy[self.datetime_col], errors='coerce')
        
        # Extract temporal features
        if 'hour' in self.extract_features:
            X_copy['transaction_hour'] = X_copy[self.datetime_col].dt.hour
            
        if 'day' in self.extract_features:
            X_copy['transaction_day'] = X_copy[self.datetime_col].dt.day
            
        if 'month' in self.extract_features:
            X_copy['transaction_month'] = X_copy[self.datetime_col].dt.month
            
        if 'year' in self.extract_features:
            X_copy['transaction_year'] = X_copy[self.datetime_col].dt.year
            
        if 'dayofweek' in self.extract_features:
            X_copy['transaction_dayofweek'] = X_copy[self.datetime_col].dt.dayofweek
            
        if 'quarter' in self.extract_features:
            X_copy['transaction_quarter'] = X_copy[self.datetime_col].dt.quarter
            
        # Additional temporal features
        X_copy['is_weekend'] = X_copy[self.datetime_col].dt.dayofweek >= 5
        X_copy['is_business_hours'] = (X_copy[self.datetime_col].dt.hour >= 9) & (X_copy[self.datetime_col].dt.hour <= 17)
        X_copy['is_late_night'] = (X_copy[self.datetime_col].dt.hour >= 22) | (X_copy[self.datetime_col].dt.hour <= 6)
        
        # Convert boolean to int
        bool_cols = ['is_weekend', 'is_business_hours', 'is_late_night']
        for col in bool_cols:
            X_copy[col] = X_copy[col].astype(int)
            
        # Fill NaN values with 0
        temporal_cols = [col for col in X_copy.columns if col.startswith('transaction_') or col.startswith('is_')]
        X_copy[temporal_cols] = X_copy[temporal_cols].fillna(0)
        
        return X_copy

def create_temporal_features(df, datetime_col='TransactionStartTime', extract_features=None):
    """Convenience function to create temporal features"""
    if extract_features is None:
        extract_features = ['hour', 'day', 'month', 'year', 'dayofweek']
    
    transformer = TemporalFeatureTransformer(datetime_col, extract_features)
    return transformer.fit_transform(df)

print("Temporal Features module loaded successfully!")
