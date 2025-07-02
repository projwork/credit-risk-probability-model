"""
Credit Risk Feature Engineering Package

A comprehensive modular feature engineering library for credit risk analysis.
Implements automated data processing using sklearn.pipeline.Pipeline with:

1. Aggregate Features: Customer and account-level aggregations
2. Temporal Features: Time-based feature extraction  
3. Categorical Encoding: One-hot and label encoding with rare category handling
4. WOE/IV Transformations: Weight of Evidence and Information Value analysis
5. Missing Values Handling: Multiple imputation strategies
6. Feature Scaling: Standard, MinMax, and Robust scaling methods

Based on credit scoring methodology and best practices for regulatory compliance.
"""

# Import all transformers
try:
    # Try relative imports first (when used as package)
    from .aggregate_features import AggregateFeatureTransformer
    from .temporal_features import TemporalFeatureTransformer
    from .categorical_encoder import CategoricalEncoderTransformer
    from .woe_iv_transformer import WOEIVTransformer
    from .missing_values_handler import MissingValuesTransformer
    from .feature_scaler import FeatureScalerTransformer
    
    # Import pipeline and utilities
    from .feature_engineering_pipeline import FeatureEngineeringPipeline, create_feature_engineering_pipeline
    
    # Import RFM and proxy target modules
    from .rfm_analyzer import RFMAnalyzer
    from .proxy_target_creator import ProxyTargetCreator
    
    # Import configuration
    from .config import *
except ImportError:
    # Fall back to absolute imports (when used directly)
    from aggregate_features import AggregateFeatureTransformer
    from temporal_features import TemporalFeatureTransformer
    from categorical_encoder import CategoricalEncoderTransformer
    from woe_iv_transformer import WOEIVTransformer
    from missing_values_handler import MissingValuesTransformer
    from feature_scaler import FeatureScalerTransformer
    
    # Import pipeline and utilities
    from feature_engineering_pipeline import FeatureEngineeringPipeline, create_feature_engineering_pipeline
    
    # Import RFM and proxy target modules
    from rfm_analyzer import RFMAnalyzer
    from proxy_target_creator import ProxyTargetCreator
    
    # Import configuration
    from config import *

# Version and metadata
__version__ = "1.0.0"
__author__ = "Credit Risk Analysis Team"
__description__ = "Modular Feature Engineering for Credit Risk Modeling"

# Define what gets imported with "from src import *"
__all__ = [
    # Core transformers
    'AggregateFeatureTransformer',
    'TemporalFeatureTransformer', 
    'CategoricalEncoderTransformer',
    'WOEIVTransformer',
    'MissingValuesTransformer',
    'FeatureScalerTransformer',
    
    # Pipeline
    'FeatureEngineeringPipeline',
    'create_feature_engineering_pipeline',
    
    # RFM and Proxy Target modules
    'RFMAnalyzer',
    'ProxyTargetCreator',
    'create_proxy_target_variable',
    
    # Configuration
    'TARGET_COLUMN',
    'RANDOM_SEED',
    'AGGREGATE_FEATURES_CONFIG',
    'WOE_IV_CONFIG'
]

def get_available_transformers():
    """Get list of available feature transformers"""
    return {
        'AggregateFeatureTransformer': 'Creates customer and account-level aggregate features',
        'TemporalFeatureTransformer': 'Extracts time-based features from datetime columns',
        'CategoricalEncoderTransformer': 'Encodes categorical variables with multiple strategies',
        'WOEIVTransformer': 'Applies Weight of Evidence transformations for credit scoring',
        'MissingValuesTransformer': 'Handles missing values with various imputation methods',
        'FeatureScalerTransformer': 'Scales numerical features using different scaling methods',
        'RFMAnalyzer': 'Calculates Recency, Frequency, Monetary metrics for customer analysis',
        'ProxyTargetCreator': 'Creates proxy target variables using RFM-based clustering'
    }

def get_pipeline_info():
    """Get information about the feature engineering pipeline"""
    return {
        'name': 'FeatureEngineeringPipeline',
        'description': 'Complete sklearn.pipeline.Pipeline for credit risk feature engineering',
        'steps': [
            'missing_values: Handle missing values and remove high-missing columns',
            'aggregate_features: Create customer and account aggregations',
            'temporal_features: Extract datetime-based features',
            'categorical_encoder: Encode categorical variables',
            'woe_iv_transform: Apply Weight of Evidence transformations',
            'feature_scaler: Scale numerical features'
        ],
        'sklearn_compatible': True,
        'supports_fit_transform': True
    }

# Print package info when imported
print("🚀 Credit Risk Feature Engineering Package loaded successfully!")
print(f"📦 Version: {__version__}")
print(f"🔧 Available transformers: {len(get_available_transformers())}")
print(f"⚙️ Pipeline steps: {len(get_pipeline_info()['steps'])}")
print("📖 Use help(src) for detailed documentation")

# Convenience function to show package capabilities
def show_capabilities():
    """Display package capabilities and usage examples"""
    print("=" * 60)
    print("CREDIT RISK FEATURE ENGINEERING CAPABILITIES")
    print("=" * 60)
    
    print("\n🔧 AVAILABLE TRANSFORMERS:")
    for name, description in get_available_transformers().items():
        print(f"  • {name}: {description}")
    
    print(f"\n⚙️ PIPELINE STEPS:")
    for i, step in enumerate(get_pipeline_info()['steps'], 1):
        print(f"  {i}. {step}")
    
    print("\n📋 QUICK USAGE EXAMPLES:")
    print("""
# Import the main pipeline
from src import create_feature_engineering_pipeline

# Load your data
import pandas as pd
df = pd.read_csv('data/raw/data.csv')

# Apply complete feature engineering pipeline
X_transformed, pipeline = create_feature_engineering_pipeline(
    df, 
    target_column='FraudResult',
    save_pipeline=True
)

# Or use individual transformers
from src import AggregateFeatureTransformer, WOEIVTransformer

# Create aggregate features
agg_transformer = AggregateFeatureTransformer()
df_with_agg = agg_transformer.fit_transform(df)

# Apply WOE transformations
woe_transformer = WOEIVTransformer(target_column='FraudResult')
df_with_woe = woe_transformer.fit_transform(df)
""")
    
    print("\n📊 WOE/IV METHODOLOGY:")
    print("  Based on: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html")
    print("  • WOE = ln(% of non-events / % of events)")
    print("  • IV = (% of non-events - % of events) * WOE")
    print("  • Used for credit scoring and risk assessment")
    
    print("\n✅ Features:")
    print("  • sklearn.pipeline.Pipeline compatible")
    print("  • Automated feature engineering workflow")
    print("  • Credit scoring methodology (WOE/IV)")
    print("  • Comprehensive data preprocessing")
    print("  • Modular and extensible design")
    print("=" * 60)

# Task 4 - Proxy Target Variable Creation Workflow
def create_proxy_target_variable(df, customer_id_col='CustomerId', date_col='TransactionStartTime', 
                                amount_col='Amount', n_clusters=3, random_state=42):
    """
    Complete workflow for creating proxy target variables using RFM analysis and clustering
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data
    customer_id_col : str
        Customer identifier column
    date_col : str
        Transaction date column
    amount_col : str
        Transaction amount column
    n_clusters : int
        Number of clusters for K-Means
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple: (data_with_target, rfm_analyzer, proxy_creator, cluster_summary)
    """
    print("🎯 Starting Proxy Target Variable Creation Workflow...")
    print("=" * 60)
    
    # Step 1: RFM Analysis
    print("Step 1: Calculating RFM metrics...")
    rfm_analyzer = RFMAnalyzer(customer_id_col=customer_id_col, 
                               date_col=date_col, 
                               amount_col=amount_col)
    rfm_analyzer.fit(df)
    
    # Step 2: Get scaled RFM features for clustering
    print("\nStep 2: Preparing features for clustering...")
    rfm_scaled = rfm_analyzer.get_scaled_rfm_features()
    
    # Step 3: Create proxy target using clustering
    print("\nStep 3: Performing customer clustering...")
    proxy_creator = ProxyTargetCreator(n_clusters=n_clusters, 
                                       random_state=random_state,
                                       customer_id_col=customer_id_col)
    proxy_creator.fit(rfm_scaled, rfm_analyzer.rfm_data)
    
    # Step 4: Apply target variable to original data
    print("\nStep 4: Creating proxy target variable...")
    data_with_target = proxy_creator.transform(df)
    
    # Step 5: Get summary
    cluster_summary = proxy_creator.get_cluster_summary()
    
    print("\n✅ Proxy Target Variable Creation Completed!")
    print(f"   High-risk customers: {cluster_summary['high_risk_count']} ({cluster_summary['high_risk_percentage']:.1f}%)")
    print("=" * 60)
    
    return data_with_target, rfm_analyzer, proxy_creator, cluster_summary
