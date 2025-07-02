"""
RFM (Recency, Frequency, Monetary) Analysis Module

This module provides comprehensive RFM analysis functionality for customer segmentation
and credit risk assessment. RFM analysis is a proven method for customer behavior analysis
that helps identify engagement patterns and risk profiles.

RFM Metrics:
- Recency: Days since last transaction (lower is better for engagement)
- Frequency: Total number of transactions (higher indicates more engagement)  
- Monetary: Total transaction value (higher indicates more valuable customer)

Based on: https://en.wikipedia.org/wiki/RFM_(market_research)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RFMAnalyzer(BaseEstimator, TransformerMixin):
    """
    RFM (Recency, Frequency, Monetary) Analysis Transformer
    
    Calculates customer engagement metrics that serve as proxies for credit risk:
    - Recency: Days since last transaction
    - Frequency: Total number of transactions  
    - Monetary: Total transaction value
    """
    
    def __init__(self, customer_id_col='CustomerId', date_col='TransactionStartTime', 
                 amount_col='Amount', snapshot_date=None):
        """
        Initialize RFM Analyzer
        
        Parameters:
        -----------
        customer_id_col : str
            Column name for customer identifier
        date_col : str  
            Column name for transaction date/time
        amount_col : str
            Column name for transaction amount
        snapshot_date : str or datetime
            Reference date for recency calculation (default: max date in data)
        """
        self.customer_id_col = customer_id_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.rfm_data = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """Fit the RFM analyzer (calculate RFM metrics)"""
        self.rfm_data = self._calculate_rfm(X)
        return self
        
    def transform(self, X):
        """Transform data by adding RFM metrics"""
        if self.rfm_data is None:
            raise ValueError("RFM analyzer not fitted. Call fit() first.")
            
        # Merge RFM data back to original dataset
        X_transformed = X.merge(
            self.rfm_data[['customer_id', 'recency', 'frequency', 'monetary']], 
            left_on=self.customer_id_col, 
            right_on='customer_id', 
            how='left'
        )
        
        return X_transformed
        
    def _calculate_rfm(self, df):
        """Calculate RFM metrics for each customer"""
        print("📊 Calculating RFM metrics...")
        
        # Ensure date column is datetime
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Set snapshot date (reference point for recency calculation)
        if self.snapshot_date is None:
            self.snapshot_date = df[self.date_col].max()
        else:
            self.snapshot_date = pd.to_datetime(self.snapshot_date)
            
        print(f"   Snapshot date: {self.snapshot_date}")
        
        # Calculate RFM metrics by customer
        rfm_metrics = df.groupby(self.customer_id_col).agg({
            self.date_col: ['max', 'count'],  # Last transaction date, transaction count
            self.amount_col: ['sum', 'mean']   # Total and average amount
        }).round(2)
        
        # Flatten column names
        rfm_metrics.columns = ['last_transaction_date', 'frequency', 'monetary_total', 'monetary_avg']
        rfm_metrics = rfm_metrics.reset_index()
        
        # Calculate recency (days since last transaction)
        rfm_metrics['recency'] = (self.snapshot_date - rfm_metrics['last_transaction_date']).dt.days
        
        # Use total monetary value as primary monetary metric
        rfm_metrics['monetary'] = rfm_metrics['monetary_total']
        
        # Add customer_id for merging
        rfm_metrics['customer_id'] = rfm_metrics[self.customer_id_col]
        
        # Add RFM scores (quintile-based scoring with fallback for limited data)
        rfm_metrics['recency_score'] = self._safe_qcut_score(rfm_metrics['recency'], reverse=True)
        rfm_metrics['frequency_score'] = self._safe_qcut_score(rfm_metrics['frequency'], reverse=False)  
        rfm_metrics['monetary_score'] = self._safe_qcut_score(rfm_metrics['monetary'], reverse=False)
        
        # Convert scores to numeric
        rfm_metrics['recency_score'] = pd.to_numeric(rfm_metrics['recency_score'], errors='coerce')
        rfm_metrics['frequency_score'] = pd.to_numeric(rfm_metrics['frequency_score'], errors='coerce')
        rfm_metrics['monetary_score'] = pd.to_numeric(rfm_metrics['monetary_score'], errors='coerce')
        
        # Calculate overall RFM score
        rfm_metrics['rfm_score'] = (
            rfm_metrics['recency_score'].fillna(1) * 100 + 
            rfm_metrics['frequency_score'].fillna(1) * 10 + 
            rfm_metrics['monetary_score'].fillna(1)
        )
        
        print(f"✅ RFM metrics calculated for {len(rfm_metrics)} customers")
        print(f"   Average Recency: {rfm_metrics['recency'].mean():.1f} days")
        print(f"   Average Frequency: {rfm_metrics['frequency'].mean():.1f} transactions")
        print(f"   Average Monetary: ${rfm_metrics['monetary'].mean():.2f}")
        
        return rfm_metrics
        
    def get_rfm_summary(self):
        """Get summary statistics of RFM metrics"""
        if self.rfm_data is None:
            raise ValueError("RFM analyzer not fitted. Call fit() first.")
            
        summary = {
            'total_customers': len(self.rfm_data),
            'recency_stats': self.rfm_data['recency'].describe(),
            'frequency_stats': self.rfm_data['frequency'].describe(), 
            'monetary_stats': self.rfm_data['monetary'].describe(),
            'rfm_scores': self.rfm_data[['recency_score', 'frequency_score', 'monetary_score']].describe()
        }
        
        return summary
        
    def get_scaled_rfm_features(self):
        """Get standardized RFM features for clustering"""
        if self.rfm_data is None:
            raise ValueError("RFM analyzer not fitted. Call fit() first.")
            
        # Prepare features for clustering (standardize them)
        rfm_features = self.rfm_data[['recency', 'frequency', 'monetary']].copy()
        
        # Handle any missing values
        rfm_features = rfm_features.fillna(rfm_features.median())
        
        # Standardize features
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Return as DataFrame with proper column names
        rfm_scaled_df = pd.DataFrame(
            rfm_scaled, 
            columns=['recency_scaled', 'frequency_scaled', 'monetary_scaled'],
            index=self.rfm_data.index
        )
        
        # Add customer_id for reference
        rfm_scaled_df['customer_id'] = self.rfm_data['customer_id'].values
        
        return rfm_scaled_df
        
    def _safe_qcut_score(self, series, reverse=False):
        """
        Safely create quantile-based scores with fallback for limited data
        
        Parameters:
        -----------
        series : pandas.Series
            The data to score
        reverse : bool
            If True, lower values get higher scores (for recency)
            If False, higher values get higher scores (for frequency/monetary)
        """
        try:
            # First try with 5 quantiles (quintiles)
            unique_vals = series.nunique()
            
            if unique_vals >= 5:
                # Try 5 quantiles
                labels = [5,4,3,2,1] if reverse else [1,2,3,4,5]
                return pd.qcut(series, q=5, labels=labels, duplicates='drop')
            elif unique_vals >= 3:
                # Try 3 quantiles (tertiles)
                labels = [3,2,1] if reverse else [1,2,3]
                return pd.qcut(series, q=3, labels=labels, duplicates='drop')
            elif unique_vals >= 2:
                # Try 2 quantiles (median split)
                labels = [2,1] if reverse else [1,2]
                return pd.qcut(series, q=2, labels=labels, duplicates='drop')
            else:
                # All values are the same, assign middle score
                return pd.Series([3] * len(series), index=series.index)
                
        except Exception:
            # Fallback: use ranking-based approach
            try:
                if reverse:
                    # For recency: lower values (more recent) get higher scores
                    scores = pd.qcut(series.rank(method='dense', ascending=True), 
                                   q=min(5, series.nunique()), 
                                   labels=list(range(5, 5-min(5, series.nunique()), -1)),
                                   duplicates='drop')
                else:
                    # For frequency/monetary: higher values get higher scores
                    scores = pd.qcut(series.rank(method='dense', ascending=False), 
                                   q=min(5, series.nunique()), 
                                   labels=list(range(5, 5-min(5, series.nunique()), -1)),
                                   duplicates='drop')
                return scores
            except Exception:
                # Final fallback: assign middle score to all
                return pd.Series([3] * len(series), index=series.index)

print("📊 RFM Analyzer module loaded!") 