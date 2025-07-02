
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class FeatureScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaling_method='standard', exclude_columns=None):
        self.scaling_method = scaling_method
        self.exclude_columns = exclude_columns or []
        self.scaler = None
        self.feature_columns = []
        
    def fit(self, X, y=None):
        # Get numerical columns to scale
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numerical_cols if col not in self.exclude_columns]
        
        if not self.feature_columns:
            return self
            
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f'Unknown scaling method: {self.scaling_method}')
            
        # Fit scaler
        self.scaler.fit(X[self.feature_columns])
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        if self.scaler and self.feature_columns:
            X_copy[self.feature_columns] = self.scaler.transform(X_copy[self.feature_columns])
            
        return X_copy

print('Feature Scaler Transformer module loaded!')
