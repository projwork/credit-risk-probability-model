"""
Categorical Encoding Module for Credit Risk Analysis
Handles encoding of categorical variables using various techniques
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    """Encodes categorical variables using multiple techniques"""
    
    def __init__(self, onehot_columns=None, label_encode_columns=None, 
                 high_cardinality_threshold=50, rare_category_threshold=0.01):
        self.onehot_columns = onehot_columns or []
        self.label_encode_columns = label_encode_columns or []
        self.high_cardinality_threshold = high_cardinality_threshold
        self.rare_category_threshold = rare_category_threshold
        self.label_encoders = {}
        self.onehot_encoder = None
        self.rare_categories = {}
        
    def fit(self, X, y=None):
        # Identify categorical columns if not specified
        if not self.onehot_columns and not self.label_encode_columns:
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            # Split based on cardinality
            for col in categorical_cols:
                if X[col].nunique() <= 10:
                    self.onehot_columns.append(col)
                else:
                    self.label_encode_columns.append(col)
        
        # Handle rare categories
        for col in self.onehot_columns + self.label_encode_columns:
            if col in X.columns:
                value_counts = X[col].value_counts(normalize=True)
                rare_cats = value_counts[value_counts < self.rare_category_threshold].index.tolist()
                self.rare_categories[col] = rare_cats
        
        # Fit label encoders
        for col in self.label_encode_columns:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                # Handle rare categories
                X_temp = X[col].copy()
                if col in self.rare_categories:
                    X_temp = X_temp.replace(self.rare_categories[col], 'RARE_CATEGORY')
                self.label_encoders[col].fit(X_temp.fillna('MISSING'))
        
        # Fit one-hot encoder
        if self.onehot_columns:
            onehot_cols_present = [col for col in self.onehot_columns if col in X.columns]
            if onehot_cols_present:
                X_onehot = X[onehot_cols_present].copy()
                # Handle rare categories
                for col in onehot_cols_present:
                    if col in self.rare_categories:
                        X_onehot[col] = X_onehot[col].replace(self.rare_categories[col], 'RARE_CATEGORY')
                
                X_onehot = X_onehot.fillna('MISSING')
                self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                self.onehot_encoder.fit(X_onehot)
        
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        # Apply label encoding
        for col, encoder in self.label_encoders.items():
            if col in X_copy.columns:
                X_temp = X_copy[col].copy()
                # Handle rare categories
                if col in self.rare_categories:
                    X_temp = X_temp.replace(self.rare_categories[col], 'RARE_CATEGORY')
                X_temp = X_temp.fillna('MISSING')
                
                # Handle unknown categories
                known_classes = set(encoder.classes_)
                X_temp = X_temp.apply(lambda x: x if x in known_classes else 'MISSING')
                
                X_copy[col + '_encoded'] = encoder.transform(X_temp)
        
        # Apply one-hot encoding
        if self.onehot_encoder is not None:
            onehot_cols_present = [col for col in self.onehot_columns if col in X_copy.columns]
            if onehot_cols_present:
                X_onehot = X_copy[onehot_cols_present].copy()
                # Handle rare categories
                for col in onehot_cols_present:
                    if col in self.rare_categories:
                        X_onehot[col] = X_onehot[col].replace(self.rare_categories[col], 'RARE_CATEGORY')
                
                X_onehot = X_onehot.fillna('MISSING')
                
                # Transform
                onehot_encoded = self.onehot_encoder.transform(X_onehot)
                feature_names = self.onehot_encoder.get_feature_names_out(onehot_cols_present)
                
                # Add to dataframe
                onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=X_copy.index)
                X_copy = pd.concat([X_copy, onehot_df], axis=1)
        
        return X_copy

def encode_categorical_features(df, onehot_columns=None, label_encode_columns=None):
    """Convenience function to encode categorical features"""
    transformer = CategoricalEncoderTransformer(onehot_columns, label_encode_columns)
    return transformer.fit_transform(df)

print("Categorical Encoder module loaded successfully!")
