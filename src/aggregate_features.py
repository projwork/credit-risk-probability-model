
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
 