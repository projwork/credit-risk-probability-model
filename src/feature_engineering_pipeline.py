
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import transformers
try:
    # Try relative imports first (when used as package)
    from .aggregate_features import AggregateFeatureTransformer
    from .temporal_features import TemporalFeatureTransformer
    from .categorical_encoder import CategoricalEncoderTransformer
    from .woe_iv_transformer import WOEIVTransformer
    from .missing_values_handler import MissingValuesTransformer
    from .feature_scaler import FeatureScalerTransformer
except ImportError:
    # Fall back to absolute imports (when used directly)
    from aggregate_features import AggregateFeatureTransformer
    from temporal_features import TemporalFeatureTransformer
    from categorical_encoder import CategoricalEncoderTransformer
    from woe_iv_transformer import WOEIVTransformer
    from missing_values_handler import MissingValuesTransformer
    from feature_scaler import FeatureScalerTransformer

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
 