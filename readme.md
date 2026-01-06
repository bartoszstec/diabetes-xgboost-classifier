# Project Overview
A machine learning model for predicting diabetes based on biometric blood-test data from Kaggle.

# Main assumption:
- Algorithm: XGBoost (gradient boosting decision trees)
- Loss function: Binary cross-entropy
- Environment: Research/development only (not planning web interface)
- Dataset: [health test by blood dataset - Kaggle](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset)

# Project Structure (planned):
- data/ - dataset (not included in repo)
- src/ - model training scripts
- models/ - saved models
## Goals
- Build a clean, reproducible ML workflow
- Evaluate baseline and optimized XGBoost model
- Analyze which biometric factors contribute most to prediction

## Conclusions from the project:
- In medical classification tasks such as diabetes prediction, overall accuracy is not the most important evaluation metric. A balanced trade-off between precision and recall for each class is crucial, with particular emphasis on recall for the positive class (ill patients). Misclassifying an ill patient as healthy (false negative) is the most critical error, therefore the model should prioritize achieving a high recall for the positive class.
- The use of Label Encoding and One-Hot Encoding for the binary Gender feature did not affect the model’s performance. This behavior is consistent with the properties of tree-based algorithms such as XGBoost, which are insensitive to monotonic transformations of categorical variables. Feature importance analysis further confirmed that gender has only a marginal predictive contribution compared to clinical and biometric features.
- Exploratory data analysis revealed a subset of samples containing identical constant values across multiple independent laboratory features, which resulted in perfect separation of the positive class. This pattern represents severe data leakage, where the target variable is implicitly encoded in the input features, leading to artificially inflated performance metrics. Such samples were excluded from training and evaluation to ensure realistic model behavior and proper generalization.

