#!/usr/bin/env python3
"""
Script to create feature engineering modules for Credit Risk Model
"""

import os
from pathlib import Path

# Ensure src directory exists
src_dir = Path("src")
src_dir.mkdir(exist_ok=True)

# 1. Create aggregate_features.py
aggregate_content = """
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='CustomerId', account_id_col='AccountId', 
                 amount_col='Amount', value_col='Value'):
        self.customer_id_col = customer_id_col
        self.account_id_col = account_id_col
        self.amount_col = amount_col
        self.value_col = value_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        
        # Customer aggregations
        customer_aggs = X_copy.groupby(self.customer_id_col).agg({
            self.amount_col: ['count', 'sum', 'mean', 'std', 'min', 'max'],
            self.value_col: ['sum', 'mean', 'std']
        }).round(4)
        
        customer_aggs.columns = ['_'.join(col).strip() for col in customer_aggs.columns]
        customer_aggs = customer_aggs.add_prefix('customer_')
        
        X_copy = X_copy.merge(customer_aggs, left_on=self.customer_id_col, right_index=True, how='left')
        X_copy = X_copy.fillna(0)
        
        return X_copy

print("Aggregate Features module loaded!")
"""

# 2. Create temporal_features.py
temporal_content = '''"""
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
'''

# 3. Create categorical_encoder.py
categorical_content = '''"""
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
'''

# Create WOE/IV transformer
woe_content = """
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WOEIVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='FraudResult', max_bins=10):
        self.target_column = target_column
        self.max_bins = max_bins
        self.woe_mappings = {}
        self.iv_values = {}
        
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
            
        for col in numerical_cols:
            if col in X.columns:
                self._fit_feature_woe(X, col)
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for feature, woe_mapping in self.woe_mappings.items():
            if feature in X_copy.columns:
                X_copy[f'{feature}_woe'] = self._apply_woe_transform(X_copy[feature], woe_mapping)
        return X_copy
        
    def _fit_feature_woe(self, X, feature):
        try:
            feature_data = X[feature].dropna()
            bins = pd.qcut(feature_data, q=min(self.max_bins, len(feature_data.unique())), duplicates='drop')
            X_temp = X[[feature, self.target_column]].copy()
            X_temp['bin'] = pd.cut(X_temp[feature], bins=bins.cat.categories, include_lowest=True)
            
            crosstab = pd.crosstab(X_temp['bin'], X_temp[self.target_column])
            if 0 not in crosstab.columns: crosstab[0] = 0
            if 1 not in crosstab.columns: crosstab[1] = 0
            
            total_events = crosstab[1].sum()
            total_non_events = crosstab[0].sum()
            
            woe_mapping = {}
            iv_total = 0
            
            for bin_name in crosstab.index:
                events = crosstab.loc[bin_name, 1]
                non_events = crosstab.loc[bin_name, 0]
                
                pct_events = (events + 0.5) / (total_events + 0.5)
                pct_non_events = (non_events + 0.5) / (total_non_events + 0.5)
                
                woe = np.log(pct_non_events / pct_events)
                iv = (pct_non_events - pct_events) * woe
                
                woe_mapping[bin_name] = woe
                iv_total += iv
                
            self.woe_mappings[feature] = woe_mapping
            self.iv_values[feature] = iv_total
            
        except Exception as e:
            print(f'Error processing {feature}: {e}')
            self.woe_mappings[feature] = {}
            self.iv_values[feature] = 0
            
    def _apply_woe_transform(self, feature_series, woe_mapping):
        result = feature_series.copy()
        for bin_interval, woe_value in woe_mapping.items():
            if hasattr(bin_interval, 'left'):
                mask = (feature_series >= bin_interval.left) & (feature_series <= bin_interval.right)
                result.loc[mask] = woe_value
        return result.fillna(0)
        
    def get_iv_summary(self):
        return pd.DataFrame(list(self.iv_values.items()), columns=['feature', 'iv']).sort_values('iv', ascending=False)

print('WOE/IV Transformer module loaded!')
"""

# Create missing values handler
missing_content = """
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
"""

# Create feature scaler
scaler_content = """
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
"""

# Create the main pipeline
pipeline_content = """
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import transformers
from .aggregate_features import AggregateFeatureTransformer
from .temporal_features import TemporalFeatureTransformer
from .categorical_encoder import CategoricalEncoderTransformer
from .woe_iv_transformer import WOEIVTransformer
from .missing_values_handler import MissingValuesTransformer
from .feature_scaler import FeatureScalerTransformer

class FeatureEngineeringPipeline:
    def __init__(self):
        self.pipeline = None
        self.is_fitted = False
        self.transformation_summary = {}
        
    def build_pipeline(self, target_column='FraudResult'):
        # Build sklearn Pipeline with all transformation steps
        self.pipeline = Pipeline([
            ('missing_values', MissingValuesTransformer()),
            ('aggregate_features', AggregateFeatureTransformer()),
            ('temporal_features', TemporalFeatureTransformer()),
            ('categorical_encoder', CategoricalEncoderTransformer()),
            ('woe_iv_transform', WOEIVTransformer(target_column=target_column)),
            ('feature_scaler', FeatureScalerTransformer())
        ], verbose=True)
        return self.pipeline
        
    def fit(self, X, y=None):
        if self.pipeline is None:
            self.build_pipeline()
        print('Fitting Feature Engineering Pipeline...')
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self
        
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError('Pipeline not fitted. Call fit() first.')
        return self.pipeline.transform(X)
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        
    def save_pipeline(self, filepath='feature_pipeline.pkl'):
        if not self.is_fitted:
            raise ValueError('Pipeline not fitted')
        joblib.dump(self.pipeline, filepath)
        print(f'Pipeline saved to: {filepath}')
        
    def get_feature_importance(self):
        woe_transformer = None
        for name, transformer in self.pipeline.steps:
            if hasattr(transformer, 'get_iv_summary'):
                return transformer.get_iv_summary()
        return pd.DataFrame()

def create_feature_engineering_pipeline(X, target_column='FraudResult', save_pipeline=False):
    print('Creating Feature Engineering Pipeline...')
    fe_pipeline = FeatureEngineeringPipeline()
    X_transformed = fe_pipeline.fit_transform(X)
    
    if save_pipeline:
        fe_pipeline.save_pipeline('feature_engineering_pipeline.pkl')
        
    print(f'Original shape: {X.shape}')
    print(f'Transformed shape: {X_transformed.shape}')
    print(f'Features added: {X_transformed.shape[1] - X.shape[1]}')
    
    return X_transformed, fe_pipeline

print('Feature Engineering Pipeline module loaded!')
"""

# Create main application script
main_app_content = """
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from feature_engineering_pipeline import create_feature_engineering_pipeline

def main():
    print('Credit Risk Feature Engineering Application')
    print('=' * 50)
    
    # Load data
    data_path = 'data/raw/data.csv'
    if not os.path.exists(data_path):
        print(f'Data file not found: {data_path}')
        print('Please ensure data.csv is in the data/raw/ directory')
        return
        
    print(f'Loading data from: {data_path}')
    df = pd.read_csv(data_path)
    print(f'Data loaded: {df.shape}')
    
    # Apply feature engineering
    print('\\nApplying feature engineering pipeline...')
    X_transformed, pipeline = create_feature_engineering_pipeline(
        df, 
        target_column='FraudResult',
        save_pipeline=True
    )
    
    # Save transformed data
    output_path = 'data/processed/features_engineered.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_transformed.to_csv(output_path, index=False)
    print(f'\\nTransformed data saved to: {output_path}')
    
    # Get feature importance
    importance = pipeline.get_feature_importance()
    if not importance.empty:
        print('\\nTop 10 Features by Information Value:')
        print(importance.head(10))
        
        # Save feature importance
        importance_path = 'data/processed/feature_importance.csv'
        importance.to_csv(importance_path, index=False)
        print(f'\\nFeature importance saved to: {importance_path}')
    
    print('\\nFeature engineering completed successfully!')

if __name__ == '__main__':
    main()
"""

# Update the files_to_create dictionary
files_to_create = {
    'feature_engineering_pipeline.py': pipeline_content,
    'main_feature_engineering.py': main_app_content
}

# Write the new files
for filename, content in files_to_create.items():
    with open(f'src/{filename}', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Created src/{filename}')

print('Main pipeline and application scripts created!') 