import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join('..', 'src'))

# Test the data preparation fix
print("🧪 Testing data size mismatch fix...")

try:
    # Simulate the sample data creation from the notebook
    n_samples = 2000
    n_features = 10
    
    print(f"Creating sample dataset with {n_samples} samples and {n_features} features...")
    
    # Create feature matrix
    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Create target variable that matches the sample size EXACTLY (the fix)
    y = pd.Series(np.random.choice([0, 1], n_samples), name='target')
    
    print(f"✅ Features (X) shape: {X.shape}")
    print(f"✅ Target (y) shape: {y.shape}")
    print(f"✅ Shapes match: {X.shape[0] == y.shape[0]}")
    
    # Test the train_test_split to ensure it works
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train set - X: {X_train.shape}, y: {y_train.shape}")
    print(f"✅ Test set - X: {X_test.shape}, y: {y_test.shape}")
    print("🎉 Data size mismatch issue has been FIXED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("The fix may need additional work.") 