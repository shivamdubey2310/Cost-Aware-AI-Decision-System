# Cost-Aware AI Decision System

> **NexGen Financial Services loses $500K daily to fraud. With only 50 investigators, they were prioritizing by risk score alone - wasting resources on low-value cases. Our system optimizes for expected savings, capturing 55% more value with the same team.**

[![Databricks](https://img.shields.io/badge/Platform-Databricks-orange)](https://databricks.com)
[![Delta Lake](https://img.shields.io/badge/Storage-Delta%20Lake-blue)](https://delta.io)
[![MLflow](https://img.shields.io/badge/ML%20Tracking-MLflow-green)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Language-Python%203.10+-yellow)](https://python.org)

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| **Improvement over Baseline** | +55% more savings captured |
| **Daily Expected Savings** | ~$198,000 (vs $127,000 baseline) |
| **Investigation Efficiency** | Optimized for ROI, not just risk |
| **Model Performance** | ROC-AUC > 0.85 |

---

## Problem Statement

### The Business Challenge

**Global fraud losses exceed $32 billion annually**, yet most fraud detection systems optimize for the *wrong objective*: prediction accuracy.

Consider this scenario at **NexGen Financial Services** (fictional bank):
- **10,000 fraud alerts** generated daily
- Only **50 investigators** available (capacity constraint)
- Each investigation costs **$75-150** in labor
- Missed fraud losses range from **$500 to $50,000+**

**Traditional Approach**: Investigate cases with highest fraud probability
- Ignores investigation cost
- Ignores potential loss magnitude
- Wastes resources on low-value high-probability cases

**Our Approach**: Optimize for **expected financial impact**
- Maximizes savings per investigation dollar spent
- Accounts for operational capacity constraints
- Prioritizes by business value, not just risk score

---

## Why AI? (Not Just Rules)

| Approach | Limitation |
|----------|------------|
| **Rule-based thresholds** | Cannot adapt to evolving fraud patterns |
| **Simple risk scoring** | Ignores cost-benefit tradeoffs |
| **Black-box ML** | Optimizes accuracy, not business outcomes |

**Our AI Innovation**: A **cost-sensitive decision system** that:
1. Predicts fraud probability using ML
2. Estimates expected financial loss if missed
3. Optimizes investigation allocation under capacity constraints
4. Outputs **actionable decisions**, not just scores

---

## Architecture: Medallion Data Pipeline

![Diagram](/diagrams/Architecture_diagram.png)

---

## Key Innovation: Business-Aware Optimization

### Traditional ML vs. Our Approach

| Metric | Traditional ML | Cost-Aware System |
|--------|---------------|-------------------|
| **Objective** | Maximize AUC/Accuracy | Maximize Expected Savings |
| **Output** | Risk probability | Investigation decision (Yes/No) |
| **Constraint** | None | Daily investigation capacity |
| **Decision Logic** | Threshold on probability | Rank by `E[savings] = P(fraud) × loss - cost` |

### The Core Formula

```
Expected Savings if Investigated = E[Loss if Ignored] - Investigation Cost

Where:
  E[Loss if Ignored] = Risk Probability × Fraud Loss if Missed
```

**We investigate cases where expected savings are highest, up to daily capacity.**

---

## Business Impact (Sample Results)

| Metric | Baseline Strategy | Cost-Aware System | Improvement |
|--------|------------------|-------------------|-------------|
| **Daily Investigations** | 50 | 50 | — |
| **Cases Correctly Prioritized** | 62% | 89% | +43% |
| **Expected Savings Captured** | $127,450 | $198,320 | **+55%** |
| **False Positive Cost** | $3,750 | $1,850 | -51% |
| **ROI per Investigation** | $2,549 | $3,966 | +56% |

> *Results based on synthetic dataset. Actual results will vary.*

---

## Project Structure

```
Cost-Aware-AI-Decision-System/
│
├── Setup.ipynb                          # Initialize catalog, schema, volumes
├── 01_Bronze_Ingestion.ipynb            # Raw data → Bronze Delta table
├── 02_Silver_Feature_Engineering.ipynb  # Feature engineering & transformations
├── 03_ML_Risk_Prediction.ipynb          # ML training with MLflow tracking
├── 04_Cost_Aware_Decision_Logic.ipynb   # Business optimization logic
├── 05_Gold_Analytics_and_Insights.ipynb # Business metrics & ROI analysis
├── 06_Interactive_Dashboard.ipynb       # Visualizations & executive summary
│
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

---

## Quick Start

### Prerequisites

- **Databricks Workspace** (Community Edition works!)
- **Unity Catalog** enabled
- Python 3.10+

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Cost-Aware-AI-Decision-System.git
   ```

2. **Upload to Databricks**
   - Import all `.ipynb` files to your Databricks workspace

3. **Run Setup notebook**
   - Execute `Setup.ipynb` to create catalog, schema, and volumes

4. **Upload sample data**
   - Upload `cost_aware_cases.csv` to the Unity Catalog Volume
   - Path: `/Volumes/cost_aware_capstone/risk_decisioning/raw_data/`

5. **Execute notebooks in order**
   ```
   Setup → 01_Bronze → 02_Silver → 03_ML → 04_Decision → 05_Analytics → 06_Dashboard
   ```

### Sample Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `case_id` | STRING | Unique case identifier |
| `transaction_amount` | DOUBLE | Transaction value ($) |
| `tx_velocity_24h` | INT | Transactions in last 24h |
| `unusual_location_flag` | INT | 1 if unusual location |
| `device_change_flag` | INT | 1 if new device |
| `account_age_days` | INT | Account age in days |
| `investigation_cost` | DOUBLE | Cost to investigate ($) |
| `fraud_loss_if_missed` | DOUBLE | Potential loss if fraud ($) |
| `label_fraud` | INT | Ground truth (1=fraud) |

---

## Technical Approach

### Feature Engineering (Silver Layer)

- **Log-scaled monetary features**: Stabilizes ML training
- **Behavioral risk score**: Composite of velocity, location, device signals
- **Expected loss proxy**: Pre-ML estimate of potential loss

### ML Model Selection

**Model**: Logistic Regression
- **Why**: Interpretable, calibrated probabilities, fast training
- **Trade-off**: Simpler model, but outputs are *decision-ready*
- **Alternative considered**: Random Forest (higher accuracy, but less calibrated)

### Decision Optimization

**Optimization Problem**:
```
Maximize: Σ (expected_savings[i] × decision[i])
Subject to: Σ decision[i] ≤ DAILY_CAPACITY
```

**Solution**: Greedy ranking by expected savings (optimal for this formulation)

---

## Assumptions & Limitations

### Assumptions
1. Investigation cost is known upfront
2. Fraud loss estimates are reliable
3. Single investigation resolves the case
4. No dependencies between cases

### Limitations
1. Model assumes stationary fraud patterns
2. Capacity is treated as fixed (no dynamic allocation)
3. Customer experience impact not modeled
4. No real-time streaming component (batch only)

### Future Enhancements
- Multi-objective optimization (cost + customer satisfaction)
- Dynamic capacity allocation by risk tier
- Real-time scoring with Structured Streaming
- A/B testing framework for strategy comparison

---

## Contest Alignment

This project addresses all evaluation criteria:

| Criterion | How We Address It |
|-----------|-------------------|
| **Problem Definition** | Clear fraud decisioning objective with defined I/O |
| **Data Understanding** | Comprehensive feature engineering with business logic |
| **AI Innovation** | Cost-aware optimization, not just prediction |
| **Model Selection** | Justified choice of Logistic Regression |
| **Training & Evaluation** | MLflow tracking, proper splits, business metrics |
| **Database ↔ AI Workflow** | End-to-end Delta Lake pipeline with Unity Catalog |
| **Business Impact** | Quantified ROI, actionable recommendations |
| **Documentation** | Clear structure, assumptions stated, reproducible |

---

## References

- [Delta Lake Documentation](https://docs.delta.io/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

## Team

- **Project by**: Shivam Dubey
- **Domain**: Finance & Banking / Risk Management
- **Tools**: Databricks, Delta Lake, MLflow, PySpark

---

## Acknowledgments

Special thanks to the contest organizers - **Indian Data Club, Databricks and Codebasics** for creating this opportunity to explore the intersection of machine learning and business decision optimization. This project was built to demonstrate how AI can deliver measurable financial impact, not just prediction accuracy.

---

## License

This project is for educational and competition purposes.

---

<p align="center">
  <b>Optimizing for business impact, not just prediction accuracy</b>
</p>