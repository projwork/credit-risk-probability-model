
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
    print('\nApplying feature engineering pipeline...')
    X_transformed, pipeline = create_feature_engineering_pipeline(
        df, 
        target_column='FraudResult',
        save_pipeline=True
    )
    
    # Save transformed data
    output_path = 'data/processed/features_engineered.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_transformed.to_csv(output_path, index=False)
    print(f'\nTransformed data saved to: {output_path}')
    
    # Get feature importance
    importance = pipeline.get_feature_importance()
    if not importance.empty:
        print('\nTop 10 Features by Information Value:')
        print(importance.head(10))
        
        # Save feature importance
        importance_path = 'data/processed/feature_importance.csv'
        importance.to_csv(importance_path, index=False)
        print(f'\nFeature importance saved to: {importance_path}')
    
    print('\nFeature engineering completed successfully!')

if __name__ == '__main__':
    main()
