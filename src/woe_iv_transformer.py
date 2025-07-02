
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
 