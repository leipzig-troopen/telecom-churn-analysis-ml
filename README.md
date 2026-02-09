# Telco Customer Churn Prediction

## Project Overview
This project develops a high-sensitivity machine learning pipeline to identify customers at risk of churn for a telecommunications provider. By shifting focus from standard accuracy to high-recall modeling, the system ensures that approximately 82% of actual churners are flagged for proactive retention efforts.

## Project Structure
The repository is organized into a modular architecture to ensure scalability and maintainability:
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for EDA, experimental modeling, and extracting actionable business insights to reduce customer attrition.
- `src/`: Core logic including data preprocessing, visualization, and model training scripts.
- `requirements.txt`: List of dependencies required to run the environment.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Identified significant class imbalance (approx. 26% churn rate).
- Analyzed key correlations between `tenure`, `Contract`, `InternetService`, and churn probability.

### 2. Modeling & Strategy
- **Baseline**: Logistic Regression used to establish a linear performance floor.
- **Candidate Models**: Random Forest and XGBoost selected for their ability to handle non-linear relationships.
- **Optimization**: Conducted exhaustive Hyperparameter Tuning via `GridSearchCV`. 
- **Imbalance Handling**: Utilized `scale_pos_weight` and `class_weight` to prioritize Recall over standard Accuracy.

### 3. Model Interpretation (SHAP)
Utilized SHapley Additive exPlanations to provide transparency into model decisions. Key churn drivers include:
- **Contract Type**: Month-to-month contracts are the highest risk factor.
- **Tenure**: Lower tenure strongly correlates with increased churn risk.
- **Service Tier**: Fiber Optic users exhibit higher attrition rates.

## Results

| Model | Recall (Churn) | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: |
| **Optimized XGBoost** | **0.82** | 0.62 | **0.844** |
| Optimized Random Forest | 0.76 | 0.63 | 0.844 |
| Logistic Regression | 0.55 | 0.59 | 0.839 |



## Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the analysis: Execute the main notebook or training scripts in `src/`.
