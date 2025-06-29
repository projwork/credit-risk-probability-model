# Credit Risk Probability Model

## Project Overview

This project focuses on developing a credit risk assessment model using transaction data to predict the probability of credit default. The model aims to provide interpretable and reliable risk scoring for financial decision-making in compliance with regulatory standards.

## Data Description

The dataset contains transaction-level information with the following key features:

- **TransactionId**: Unique transaction identifier on platform
- **BatchId**: Unique number assigned to a batch of transactions for processing
- **AccountId**: Unique number identifying the customer on platform
- **SubscriptionId**: Unique number identifying the customer subscription
- **CustomerId**: Unique identifier attached to Account
- **CurrencyCode**: Country currency
- **CountryCode**: Numerical geographical code of country
- **ProviderId**: Source provider of Item bought
- **ProductId**: Item name being bought
- **ProductCategory**: ProductIds are organized into these broader product categories
- **ChannelId**: Identifies if customer used web, Android, iOS, pay later or checkout
- **Amount**: Value of the transaction (Positive for debits, negative for credits)
- **Value**: Absolute value of the amount
- **TransactionStartTime**: Transaction start time
- **PricingStrategy**: Category of Xente's pricing structure for merchants
- **FraudResult**: Fraud status of transaction (1 = yes, 0 = no)

_Data Location: `/data/raw/` folder_

## Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Interpretability and Documentation

The Basel II Accord fundamentally transformed credit risk management by establishing stringent requirements for risk measurement and capital adequacy. Its emphasis on risk measurement directly influences our modeling approach in several critical ways:

**Regulatory Compliance Requirements:**

- **Model Validation**: Basel II mandates that financial institutions must validate their internal rating systems, requiring transparent and interpretable models that can be audited by regulators
- **Documentation Standards**: The accord requires comprehensive documentation of model development, validation, and ongoing monitoring processes
- **Backtesting and Stress Testing**: Models must demonstrate consistent performance over time and under various economic scenarios, necessitating clear model logic that can be easily tested and validated

**Risk Management Framework:**

- **Three Pillars Approach**: The accord's three-pillar structure (minimum capital requirements, supervisory review, and market discipline) demands models that can provide clear risk assessments that stakeholders can understand and act upon
- **Internal Ratings-Based (IRB) Approach**: For institutions using advanced IRB approaches, models must meet strict statistical and conceptual soundness requirements, emphasizing the need for interpretable features and transparent methodology

**Business Implications:**

- Models must balance statistical accuracy with regulatory acceptability
- Clear audit trails and explainability are not optional but mandatory
- Risk officers and regulators must be able to understand and challenge model outputs
- Model governance frameworks must ensure ongoing compliance and performance monitoring

### 2. Proxy Variable Necessity and Associated Business Risks

In the absence of direct default labels in our transaction dataset, creating proxy variables becomes essential for credit risk modeling, though this approach introduces significant considerations:

**Why Proxy Variables Are Necessary:**

- **Data Limitations**: Our dataset lacks explicit default indicators, making it impossible to directly model traditional credit risk outcomes
- **Predictive Modeling Requirements**: Machine learning models require target variables to learn patterns associated with risk
- **Business Decision Support**: Financial institutions need risk scores to make lending decisions, even when direct default history is unavailable
- **Regulatory Expectations**: Risk management frameworks require quantitative assessments of credit risk, necessitating some form of risk measurement

**Potential Business Risks of Proxy-Based Predictions:**

**Model Risk:**

- **Proxy Validity**: The proxy may not accurately represent true default risk, leading to systematic prediction errors
- **Concept Drift**: The relationship between proxy variables and actual risk may change over time
- **False Correlations**: Apparent patterns in proxy data may not translate to real-world default behavior

**Operational Risks:**

- **Decision Quality**: Poor proxy variables can lead to suboptimal lending decisions, affecting portfolio performance
- **Customer Impact**: Incorrectly classified customers may be denied credit inappropriately or granted credit unsustainably
- **Regulatory Scrutiny**: Regulators may challenge models based on proxy variables, especially if they show bias or poor performance

**Financial Risks:**

- **Capital Allocation**: Incorrect risk assessments can lead to inadequate capital reserves or missed business opportunities
- **Portfolio Concentration**: Biased proxies might systematically favor or penalize certain customer segments
- **Stress Testing Limitations**: Proxy-based models may not accurately reflect behavior under economic stress

**Mitigation Strategies:**

- Comprehensive validation using external data sources when available
- Regular monitoring and recalibration of proxy relationships
- Conservative risk margins to account for proxy uncertainty
- Clear documentation of limitations and assumptions

### 3. Model Complexity Trade-offs in Regulated Financial Contexts

The choice between simple, interpretable models and complex, high-performance models in financial services involves critical trade-offs that extend beyond pure predictive accuracy:

**Simple, Interpretable Models (e.g., Logistic Regression with Weight of Evidence):**

**Advantages:**

- **Regulatory Compliance**: Easier to explain to regulators and pass model validation requirements
- **Transparency**: Clear understanding of how each feature contributes to risk assessment
- **Auditability**: Straightforward to audit and validate model logic and outputs
- **Stability**: Less prone to overfitting and more stable across different time periods
- **Business Intuition**: Coefficients can be easily interpreted by risk managers and business stakeholders
- **Model Governance**: Simpler to monitor, maintain, and update

**Disadvantages:**

- **Limited Complexity**: May miss complex, non-linear relationships in data
- **Feature Engineering Burden**: Requires extensive manual feature engineering (like WoE binning)
- **Performance Ceiling**: May underperform in terms of pure predictive accuracy

**Complex, High-Performance Models (e.g., Gradient Boosting, Random Forest):**

**Advantages:**

- **Superior Accuracy**: Often achieve better predictive performance and discrimination
- **Automatic Feature Interactions**: Can capture complex, non-linear relationships without manual engineering
- **Robustness**: Can handle missing values and outliers more effectively
- **Feature Selection**: Built-in mechanisms for identifying important predictors

**Disadvantages:**

- **Black Box Nature**: Difficult to explain individual predictions to regulators and stakeholders
- **Overfitting Risk**: More prone to capturing noise rather than signal, especially with limited data
- **Validation Complexity**: Harder to validate and may not meet regulatory standards for explainability
- **Maintenance Burden**: More complex to monitor, update, and debug
- **Regulatory Risk**: May face rejection by regulators due to lack of interpretability

**Optimal Approach in Regulated Contexts:**

**Hybrid Strategy:**

- Use interpretable models as the primary regulatory model for compliance and decision-making
- Employ complex models for benchmark comparison and to identify potential model improvements
- Implement model ensemble approaches that combine interpretability with performance

**Risk-Adjusted Decision Framework:**

- Prioritize interpretability for high-stakes decisions (large loans, regulatory reporting)
- Consider complex models for initial screening or portfolio-level analysis
- Maintain clear model hierarchy with interpretable models as the final arbitrator

**Regulatory Alignment:**

- Ensure chosen approach aligns with institutional risk appetite and regulatory requirements
- Establish clear model governance frameworks regardless of complexity
- Maintain comprehensive documentation and validation procedures
- Regular stakeholder communication about model limitations and assumptions

The optimal choice depends on the specific regulatory environment, institutional risk tolerance, data quality, and business objectives, with interpretability often taking precedence over marginal performance gains in highly regulated financial contexts.
