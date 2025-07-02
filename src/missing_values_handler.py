
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

class MissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='auto', numerical_strategy='median', 
                 categorical_strategy='most_frequent', missing_threshold=0.5):
        self.strategy = strategy
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.missing_threshold = missing_threshold
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.columns_to_drop = []
        
    def fit(self, X, y=None):
        # Identify columns to drop (too many missing values)
        missing_pct = X.isnull().sum() / len(X)
        self.columns_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        X_clean = X.drop(columns=self.columns_to_drop)
        
        # Fit imputers for remaining columns
        numerical_cols = X_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = X_clean.select_dtypes(include=['object']).columns
        
        if len(numerical_cols) > 0:
            self.numerical_imputer = SimpleImputer(strategy=self.numerical_strategy)
            self.numerical_imputer.fit(X_clean[numerical_cols])
            
        if len(categorical_cols) > 0:
            self.categorical_imputer = SimpleImputer(strategy=self.categorical_strategy)
            self.categorical_imputer.fit(X_clean[categorical_cols])
            
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        # Drop high missing columns
        X_copy = X_copy.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Apply imputation
        numerical_cols = X_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = X_copy.select_dtypes(include=['object']).columns
        
        if self.numerical_imputer and len(numerical_cols) > 0:
            X_copy[numerical_cols] = self.numerical_imputer.transform(X_copy[numerical_cols])
            
        if self.categorical_imputer and len(categorical_cols) > 0:
            X_copy[categorical_cols] = self.categorical_imputer.transform(X_copy[categorical_cols])
            
        return X_copy

print('Missing Values Transformer module loaded!')
