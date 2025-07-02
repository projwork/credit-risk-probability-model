# Credit Risk Feature Engineering Documentation

## Overview

This document provides comprehensive documentation for the modular feature engineering system implemented for credit risk probability modeling. The system implements a robust, automated, and reproducible data processing pipeline using `sklearn.pipeline.Pipeline` to transform raw transaction data into model-ready features.

## 🏗️ Architecture & Design

### Core Components

The feature engineering system consists of 6 specialized modules located in the `/src` directory:

1. **`config.py`** - Configuration and settings
2. **`aggregate_features.py`** - Customer and account-level aggregations
3. **`temporal_features.py`** - Time-based feature extraction
4. **`categorical_encoder.py`** - Categorical variable encoding
5. **`woe_iv_transformer.py`** - Weight of Evidence transformations
6. **`missing_values_handler.py`** - Missing value imputation
7. **`feature_scaler.py`** - Numerical feature scaling
8. **`feature_engineering_pipeline.py`** - Main pipeline orchestration

### sklearn.pipeline.Pipeline Implementation

The system uses `sklearn.pipeline.Pipeline` to chain all transformation steps:

```python
Pipeline([
    ('missing_values', MissingValuesTransformer()),
    ('aggregate_features', AggregateFeatureTransformer()),
    ('temporal_features', TemporalFeatureTransformer()),
    ('categorical_encoder', CategoricalEncoderTransformer()),
    ('woe_iv_transform', WOEIVTransformer(target_column='FraudResult')),
    ('feature_scaler', FeatureScalerTransformer())
])
```

## 📊 Feature Engineering Components

### 1. Aggregate Features

**Purpose**: Create customer-level and account-level statistical summaries

**Features Generated**:

- **Customer-level aggregations**:

  - `customer_transaction_count`: Number of transactions per customer
  - `customer_total_amount`: Sum of all transaction amounts
  - `customer_avg_amount`: Average transaction amount
  - `customer_amount_std`: Standard deviation of amounts
  - `customer_min_amount`, `customer_max_amount`: Min/max amounts
  - `customer_amount_range`: Range of transaction amounts
  - `customer_amount_cv`: Coefficient of variation

- **Account-level aggregations**:

  - `account_transaction_count`: Transactions per account
  - `account_avg_amount`: Average amount per account
  - `account_unique_customers`: Number of unique customers per account

- **Cross-dimensional features**:
  - `amount_vs_customer_avg`: Transaction amount relative to customer average
  - `amount_zscore_customer`: Z-score of transaction vs customer behavior

**Business Value**: Identifies customer behavior patterns and outlier transactions

### 2. Temporal Features

**Purpose**: Extract time-based patterns from transaction timestamps

**Features Generated**:

- `transaction_hour`: Hour of transaction (0-23)
- `transaction_day`: Day of month (1-31)
- `transaction_month`: Month (1-12)
- `transaction_year`: Year
- `transaction_dayofweek`: Day of week (0-6)
- `transaction_quarter`: Quarter (1-4)
- `is_weekend`: Boolean indicator for weekend transactions
- `is_business_hours`: Boolean for business hours (9 AM - 5 PM)
- `is_late_night`: Boolean for late night transactions (10 PM - 6 AM)

**Business Value**: Captures temporal fraud patterns and seasonal behavior

### 3. Categorical Encoding

**Purpose**: Convert categorical variables to numerical format

**Strategies**:

- **One-Hot Encoding**: For low cardinality features (≤10 unique values)
  - Creates binary columns for each category
  - Handles rare categories (threshold: 1% of data)
- **Label Encoding**: For high cardinality features (>10 unique values)
  - Assigns integer codes to categories
  - Memory efficient for many categories

**Rare Category Handling**: Categories appearing in <1% of data are grouped as 'RARE_CATEGORY'

**Business Value**: Enables machine learning algorithms to process categorical data while maintaining interpretability

### 4. Weight of Evidence (WOE) & Information Value (IV)

**Purpose**: Transform features based on their relationship with the target variable

**Methodology** (Based on [ListenData article](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)):

**Weight of Evidence (WOE)**:

```
WOE = ln(% of non-events / % of events)
```

- **Positive WOE**: Distribution of goods > Distribution of bads
- **Negative WOE**: Distribution of goods < Distribution of bads

**Information Value (IV)**:

```
IV = (% of non-events - % of events) × WOE
```

**IV Interpretation**:

- IV < 0.02: Not useful for prediction
- 0.02 ≤ IV < 0.1: Weak predictive power
- 0.1 ≤ IV < 0.3: Medium predictive power
- 0.3 ≤ IV < 0.5: Strong predictive power
- IV ≥ 0.5: Suspicious, too good to be true

**Implementation**:

- Creates optimal bins for continuous variables (max 10 bins)
- Ensures minimum 5% of data in each bin
- Applies smoothing to handle zero counts
- Generates `{feature}_woe` transformed features

**Business Value**:

- Provides interpretable risk scores
- Meets regulatory requirements for model explainability
- Optimizes predictive power while maintaining business intuition

### 5. Missing Values Handling

**Purpose**: Handle missing data appropriately for different feature types

**Strategies**:

- **Column Removal**: Drop columns with >50% missing values
- **Numerical Imputation**: Median imputation (robust to outliers)
- **Categorical Imputation**: Mode imputation (most frequent category)
- **Advanced Options**: KNN imputation available for complex patterns

**Business Value**: Ensures robust model performance and prevents data leakage

### 6. Feature Scaling

**Purpose**: Normalize numerical features to similar scales

**Methods Available**:

- **StandardScaler**: Mean=0, Std=1 (default)
- **MinMaxScaler**: Scale to [0,1] range
- **RobustScaler**: Uses median and IQR (robust to outliers)

**Business Value**: Improves model convergence and ensures fair feature importance

## 🚀 Usage Examples

### Quick Start

```python
# Import the main pipeline
from src import create_feature_engineering_pipeline
import pandas as pd

# Load your data
df = pd.read_csv('data/raw/data.csv')

# Apply complete feature engineering
X_transformed, pipeline = create_feature_engineering_pipeline(
    df,
    target_column='FraudResult',
    save_pipeline=True
)

print(f"Original shape: {df.shape}")
print(f"Transformed shape: {X_transformed.shape}")
```

### Individual Transformers

```python
from src import AggregateFeatureTransformer, WOEIVTransformer

# Create aggregate features
agg_transformer = AggregateFeatureTransformer()
df_with_agg = agg_transformer.fit_transform(df)

# Apply WOE transformations
woe_transformer = WOEIVTransformer(target_column='FraudResult')
df_with_woe = woe_transformer.fit_transform(df)

# Get feature importance
iv_summary = woe_transformer.get_iv_summary()
print("Top features by Information Value:")
print(iv_summary.head())
```

### Custom Pipeline

```python
from sklearn.pipeline import Pipeline
from src import (AggregateFeatureTransformer,
                 TemporalFeatureTransformer,
                 WOEIVTransformer)

# Build custom pipeline
custom_pipeline = Pipeline([
    ('aggregates', AggregateFeatureTransformer()),
    ('temporal', TemporalFeatureTransformer()),
    ('woe_transform', WOEIVTransformer(target_column='FraudResult'))
])

# Fit and transform
X_custom = custom_pipeline.fit_transform(df)
```

## 📁 File Structure

```
src/
├── __init__.py                      # Package initialization
├── config.py                       # Configuration settings
├── aggregate_features.py           # Customer/account aggregations
├── temporal_features.py            # Time-based features
├── categorical_encoder.py          # Categorical encoding
├── woe_iv_transformer.py          # WOE/IV transformations
├── missing_values_handler.py      # Missing value handling
├── feature_scaler.py              # Feature scaling
├── feature_engineering_pipeline.py # Main pipeline
└── main_feature_engineering.py    # Application script
```

## 🔍 Data Quality & Validation

### Built-in Checks

- Missing value analysis
- Data type validation
- Outlier detection
- Feature distribution analysis
- Target variable balance checking

### Quality Assurance

- All transformers follow sklearn BaseEstimator/TransformerMixin patterns
- Reproducible results with fixed random seeds
- Comprehensive error handling and logging
- Memory-efficient processing for large datasets

## 📈 Performance Considerations

### Scalability

- Pandas-based processing for medium datasets (<1M rows)
- Memory-efficient aggregations using groupby operations
- Lazy evaluation where possible
- Option to persist intermediate results

### Optimization

- Vectorized operations throughout
- Minimal data copying
- Efficient categorical encoding for high cardinality features
- Batch processing for WOE transformations

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Target variable
TARGET_COLUMN = 'FraudResult'

# WOE/IV settings
WOE_IV_CONFIG = {
    'max_bins': 10,
    'min_bin_size': 0.05,
    'iv_threshold': 0.02
}

# Missing values
MISSING_VALUES_CONFIG = {
    'missing_threshold': 0.5,
    'numerical_imputation': 'median',
    'categorical_imputation': 'mode'
}
```

## 🧪 Testing & Validation

### Unit Testing

- Each transformer has built-in validation
- Sample data generation for testing
- Edge case handling (empty datasets, single values, etc.)

### Integration Testing

- End-to-end pipeline testing
- Data consistency checks
- Performance benchmarking

## 📊 Output Files

The pipeline generates several output files:

1. **`data/processed/features_engineered.csv`** - Transformed dataset
2. **`data/processed/feature_importance.csv`** - IV-based feature ranking
3. **`feature_engineering_pipeline.pkl`** - Serialized pipeline for deployment
4. **Feature engineering logs** - Detailed processing logs

## 🔮 Future Enhancements

### Planned Features

- Advanced WOE binning algorithms (ChiMerge, MDLP)
- Feature selection based on IV thresholds
- Advanced temporal features (seasonality, trends)
- Integration with external data sources

### Integration Options

- MLflow integration for experiment tracking
- Apache Spark for big data processing
- Real-time streaming pipeline support

## 📚 References

1. [Weight of Evidence and Information Value Explained](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html) - ListenData
2. Basel II Accord - Risk Management Framework
3. scikit-learn Pipeline Documentation
4. Credit Risk Modeling Best Practices

## 📞 Support

For technical support or questions about the feature engineering pipeline:

1. Check the inline documentation: `help(src)`
2. Review transformer docstrings
3. Run capability overview: `src.show_capabilities()`
4. Consult the original notebook: `notebooks/creditRiskProbabilityModel.ipynb`

---

**Note**: This feature engineering system is designed for credit risk modeling and follows industry best practices for regulatory compliance and model interpretability.
