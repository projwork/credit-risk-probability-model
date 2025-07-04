# Feature Engineering Configuration
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_COLUMN = "FraudResult"
RANDOM_SEED = 42

AGGREGATE_FEATURES_CONFIG = {
    "customer_id_col": "CustomerId",
    "amount_col": "Amount", 
    "value_col": "Value"
}

WOE_IV_CONFIG = {
    "target_column": TARGET_COLUMN,
    "max_bins": 10,
    "min_bin_size": 0.05,
    "iv_threshold": 0.02
}

print("Configuration loaded!")
 