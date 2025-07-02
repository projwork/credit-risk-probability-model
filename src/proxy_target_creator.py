"""
Proxy Target Variable Creator Module

This module creates proxy target variables for credit risk modeling by analyzing
customer engagement patterns through RFM (Recency, Frequency, Monetary) clustering.
The approach identifies "disengaged" customers as high-risk proxies based on the
assumption that less engaged customers are more likely to default.

Methodology:
1. Use K-Means clustering on standardized RFM features
2. Identify the cluster with highest risk characteristics (high recency, low frequency, low monetary)
3. Create binary target variable (is_high_risk) based on cluster membership

Based on: Credit risk modeling best practices and customer engagement analysis
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ProxyTargetCreator(BaseEstimator, TransformerMixin):
    """
    Proxy Target Variable Creator using RFM-based Customer Clustering
    
    Creates a binary target variable (is_high_risk) by:
    1. Clustering customers based on RFM metrics
    2. Identifying the least engaged cluster as high-risk
    3. Assigning binary labels (1=high risk, 0=low risk)
    """
    
    def __init__(self, n_clusters=3, random_state=42, customer_id_col='CustomerId'):
        """
        Initialize Proxy Target Creator
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for K-Means (default: 3)
        random_state : int
            Random state for reproducible results
        customer_id_col : str
            Column name for customer identifier
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.customer_id_col = customer_id_col
        self.kmeans = None
        self.cluster_profiles = None
        self.high_risk_cluster = None
        self.customer_clusters = None
        
    def fit(self, rfm_scaled_features, rfm_data):
        """
        Fit the clustering model and identify high-risk cluster
        
        Parameters:
        -----------
        rfm_scaled_features : DataFrame
            Standardized RFM features for clustering
        rfm_data : DataFrame
            Original RFM data for cluster profiling
        """
        print(f"🎯 Creating proxy target variable using {self.n_clusters} clusters...")
        
        # Perform K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        
        # Fit clustering on scaled RFM features (excluding customer_id)
        clustering_features = rfm_scaled_features[['recency_scaled', 'frequency_scaled', 'monetary_scaled']]
        cluster_labels = self.kmeans.fit_predict(clustering_features)
        
        # Add cluster labels to RFM data
        self.customer_clusters = rfm_data.copy()
        self.customer_clusters['cluster'] = cluster_labels
        
        # Analyze cluster profiles
        self.cluster_profiles = self._analyze_clusters()
        
        # Identify high-risk cluster
        self.high_risk_cluster = self._identify_high_risk_cluster()
        
        print(f"✅ Clustering completed. High-risk cluster identified: {self.high_risk_cluster}")
        
        return self
        
    def transform(self, X):
        """
        Transform data by adding proxy target variable
        
        Parameters:
        -----------
        X : DataFrame
            Original dataset to add proxy target to
            
        Returns:
        --------
        DataFrame with added is_high_risk column
        """
        if self.customer_clusters is None:
            raise ValueError("Proxy target creator not fitted. Call fit() first.")
            
        # Create target variable mapping
        target_mapping = self.customer_clusters[['customer_id', 'cluster']].copy()
        target_mapping['is_high_risk'] = (target_mapping['cluster'] == self.high_risk_cluster).astype(int)
        
        # Merge with original dataset
        X_transformed = X.merge(
            target_mapping[['customer_id', 'is_high_risk']], 
            left_on=self.customer_id_col, 
            right_on='customer_id', 
            how='left'
        )
        
        # Fill any missing values with 0 (low risk)
        X_transformed['is_high_risk'] = X_transformed['is_high_risk'].fillna(0).astype(int)
        
        return X_transformed
        
    def _analyze_clusters(self):
        """Analyze cluster characteristics to understand customer segments"""
        print("📊 Analyzing cluster profiles...")
        
        cluster_analysis = self.customer_clusters.groupby('cluster').agg({
            'recency': ['mean', 'std'],
            'frequency': ['mean', 'std'], 
            'monetary': ['mean', 'std'],
            'customer_id': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_analysis.columns = [
            'recency_mean', 'recency_std',
            'frequency_mean', 'frequency_std',
            'monetary_mean', 'monetary_std',
            'customer_count'
        ]
        
        # Calculate percentages
        total_customers = cluster_analysis['customer_count'].sum()
        cluster_analysis['percentage'] = (cluster_analysis['customer_count'] / total_customers * 100).round(1)
        
        # Add risk interpretation
        cluster_analysis['risk_score'] = (
            cluster_analysis['recency_mean'] / cluster_analysis['recency_mean'].max() * 0.4 +  # Higher recency = higher risk
            (1 - cluster_analysis['frequency_mean'] / cluster_analysis['frequency_mean'].max()) * 0.3 +  # Lower frequency = higher risk
            (1 - cluster_analysis['monetary_mean'] / cluster_analysis['monetary_mean'].max()) * 0.3  # Lower monetary = higher risk
        )
        
        print("   Cluster profiles created successfully!")
        
        return cluster_analysis
        
    def _identify_high_risk_cluster(self):
        """Identify the cluster with highest risk characteristics"""
        # High-risk cluster typically has:
        # - High recency (long time since last transaction)
        # - Low frequency (few transactions)
        # - Low monetary value (low spending)
        
        high_risk_cluster_id = self.cluster_profiles['risk_score'].idxmax()
        
        print(f"   High-risk cluster: {high_risk_cluster_id}")
        print(f"   Risk characteristics:")
        profile = self.cluster_profiles.loc[high_risk_cluster_id]
        print(f"     • Average Recency: {profile['recency_mean']:.1f} days")
        print(f"     • Average Frequency: {profile['frequency_mean']:.1f} transactions") 
        print(f"     • Average Monetary: ${profile['monetary_mean']:.2f}")
        print(f"     • Customer Count: {profile['customer_count']} ({profile['percentage']:.1f}%)")
        
        return high_risk_cluster_id
        
    def get_cluster_summary(self):
        """Get detailed cluster analysis summary"""
        if self.cluster_profiles is None:
            raise ValueError("Proxy target creator not fitted. Call fit() first.")
            
        summary = {
            'cluster_profiles': self.cluster_profiles,
            'high_risk_cluster': self.high_risk_cluster,
            'total_customers': len(self.customer_clusters),
            'high_risk_count': len(self.customer_clusters[self.customer_clusters['cluster'] == self.high_risk_cluster]),
            'high_risk_percentage': len(self.customer_clusters[self.customer_clusters['cluster'] == self.high_risk_cluster]) / len(self.customer_clusters) * 100
        }
        
        return summary
        
    def plot_clusters(self, figsize=(15, 10)):
        """Visualize customer clusters with RFM analysis"""
        if self.customer_clusters is None:
            raise ValueError("Proxy target creator not fitted. Call fit() first.")
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Color map for clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple'][:self.n_clusters]
        cluster_colors = {i: colors[i] for i in range(self.n_clusters)}
        
        # Plot 1: Recency vs Frequency
        axes[0, 0].scatter(self.customer_clusters['frequency'], self.customer_clusters['recency'], 
                          c=self.customer_clusters['cluster'].map(cluster_colors), alpha=0.6)
        axes[0, 0].set_xlabel('Frequency (Transactions)')
        axes[0, 0].set_ylabel('Recency (Days)')
        axes[0, 0].set_title('Customer Clusters: Recency vs Frequency')
        
        # Plot 2: Frequency vs Monetary
        axes[0, 1].scatter(self.customer_clusters['frequency'], self.customer_clusters['monetary'], 
                          c=self.customer_clusters['cluster'].map(cluster_colors), alpha=0.6)
        axes[0, 1].set_xlabel('Frequency (Transactions)')
        axes[0, 1].set_ylabel('Monetary ($)')
        axes[0, 1].set_title('Customer Clusters: Frequency vs Monetary')
        
        # Plot 3: Recency vs Monetary
        axes[0, 2].scatter(self.customer_clusters['recency'], self.customer_clusters['monetary'], 
                          c=self.customer_clusters['cluster'].map(cluster_colors), alpha=0.6)
        axes[0, 2].set_xlabel('Recency (Days)')
        axes[0, 2].set_ylabel('Monetary ($)')
        axes[0, 2].set_title('Customer Clusters: Recency vs Monetary')
        
        # Plot 4: Cluster distribution
        cluster_counts = self.customer_clusters['cluster'].value_counts().sort_index()
        bars = axes[1, 0].bar(cluster_counts.index, cluster_counts.values, 
                              color=[cluster_colors[i] for i in cluster_counts.index], alpha=0.7)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Customer Count')
        axes[1, 0].set_title('Customer Distribution by Cluster')
        
        # Highlight high-risk cluster
        if self.high_risk_cluster is not None:
            bars[self.high_risk_cluster].set_color('red')
            bars[self.high_risk_cluster].set_alpha(1.0)
            axes[1, 0].text(self.high_risk_cluster, cluster_counts[self.high_risk_cluster] + 5, 
                           'HIGH RISK', ha='center', fontweight='bold', color='red')
        
        # Plot 5: RFM Heatmap by cluster
        cluster_means = self.customer_clusters.groupby('cluster')[['recency', 'frequency', 'monetary']].mean()
        sns.heatmap(cluster_means.T, annot=True, fmt='.1f', ax=axes[1, 1], cmap='RdYlBu_r')
        axes[1, 1].set_title('Cluster RFM Profiles Heatmap')
        axes[1, 1].set_ylabel('RFM Metrics')
        
        # Plot 6: Risk scores by cluster
        risk_scores = self.cluster_profiles['risk_score']
        bars = axes[1, 2].bar(risk_scores.index, risk_scores.values, 
                              color=[cluster_colors[i] for i in risk_scores.index], alpha=0.7)
        axes[1, 2].set_xlabel('Cluster')
        axes[1, 2].set_ylabel('Risk Score')
        axes[1, 2].set_title('Risk Score by Cluster')
        
        # Highlight high-risk cluster
        if self.high_risk_cluster is not None:
            bars[self.high_risk_cluster].set_color('red')
            bars[self.high_risk_cluster].set_alpha(1.0)
            axes[1, 2].text(self.high_risk_cluster, risk_scores[self.high_risk_cluster] + 0.02, 
                           'HIGH RISK', ha='center', fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.show()
        
    def get_target_variable_stats(self):
        """Get statistics about the created target variable"""
        if self.customer_clusters is None:
            raise ValueError("Proxy target creator not fitted. Call fit() first.")
            
        # Create target variable
        target_data = self.customer_clusters.copy()
        target_data['is_high_risk'] = (target_data['cluster'] == self.high_risk_cluster).astype(int)
        
        stats = {
            'total_customers': len(target_data),
            'high_risk_count': target_data['is_high_risk'].sum(),
            'high_risk_percentage': target_data['is_high_risk'].mean() * 100,
            'low_risk_count': len(target_data) - target_data['is_high_risk'].sum(),
            'low_risk_percentage': (1 - target_data['is_high_risk'].mean()) * 100
        }
        
        return stats

print("🎯 Proxy Target Creator module loaded!") 