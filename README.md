# 🏥 Analyze the Use of Medical Services by the Elderly

**Authors:** Pedro Kauan Silveira Silva  
**Advisor:** Bruno Riccelli dos Santos Silva

A compact, reproducible project that explores patterns of medical service usage among people aged **50+** using the **National Poll on Healthy Aging (NPHA)** dataset. We compare multiple machine learning classifiers (MLP, Decision Tree, Random Forest, SVM, XGBoost, KNN, Logistic Regression) using stratified cross-validation, GridSearch hyperparameter tuning and statistical validation (Wilcoxon test).

---

## 🚀 Quick Highlights
- Predicts annual doctor-visit frequency (categorized: `0–1`, `2–3`, `4+` visits)  
- Uses **stratified cross-validation** + **GridSearch**  
- Evaluated using **Accuracy** and **macro F1-score**  
- **Wilcoxon Test** applied for statistical validation  
- Full exploratory and ML analysis inside: `AUSMI.ipynb`  
- Dataset included: `NPHA-doctor-visits.csv`

---


## 🔬 Project Summary
Population aging increases chronic illnesses and healthcare demand. This project investigates which demographic and health factors influence the number of annual doctor visits among adults aged 50+. The goal is to provide interpretable insights and a reproducible ML benchmark for academic and policy-related decisions.

## 🧾 Dataset
**Source:** National Poll on Healthy Aging (NPHA)

**Samples:** 714

**Features:** 14 demographic and health-related variables

**Target categories:**
- 0–1 visits
- 2–3 visits
- 4+ visits

## ⚙️ Methodology Overview
- **Data Cleaning:** Removal of redundant or inconsistent features
- **Encoding:** One-Hot Encoding + ordinal labels
- **Scaling:** MinMaxScaler inside the CV pipeline
- **Splitting:** Stratified train/test split
- **Models Tested:**
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Logistic Regression
  - XGBoost
- **Validation:** 10-fold Stratified Cross-Validation
- **Hyperparameter Tuning:** GridSearchCV
- **Metrics:** Accuracy, Precision, Recall, F1-score
- **Statistical Test:** Wilcoxon signed-rank test

## 🤖 Machine Learning Models Evaluated

| Model | Hyperparameter Tuning Strategy | Evaluation Metric |
|-------|--------------------------------|-------------------|
| **MLP Classifier** | hidden_layer_sizes, activation, solver | Accuracy, F1-score |
| **Logistic Regression** | C, penalty, solver | Accuracy, F1-score |
| **Random Forest** | n_estimators, max_depth | Accuracy, F1-score |
| **Decision Tree** | max_depth, criterion | Accuracy, F1-score |
| **XGBoost** | n_estimators, max_depth, learning_rate | Accuracy, F1-score |
| **SVM** | C, kernel, gamma | Accuracy, F1-score |
| **KNN** | n_neighbors, metric | Accuracy, F1-score |

The models were evaluated using 10-fold stratified cross-validation and results showed that **MLP Classifier** achieved the highest accuracy, while **Decision Tree** obtained the best F1-score.

## 📊 Results
**Mean ± Standard Deviation across 10-fold CV**

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| MLP Classifier | 0.5240 ± 0.0560 | 0.3957 ± 0.0794 |
| Logistic Regression | 0.5015 ± 0.0724 | 0.3932 ± 0.0834 |
| Random Forest | 0.4861 ± 0.0585 | 0.4013 ± 0.0631 |
| Decision Tree | 0.4593 ± 0.0553 | 0.4199 ± 0.0575 |
| XGBoost | 0.4245 ± 0.0698 | 0.4099 ± 0.0726 |
| SVM | 0.4077 ± 0.0675 | 0.4132 ± 0.0693 |
| KNN | 0.4019 ± 0.0486 | 0.4039 ± 0.0421 |

- **Cross-Validation:** Each model was trained and evaluated on 10 folds, ensuring robust performance.
- **Statistical Test:** The Wilcoxon test confirmed statistically significant differences between models, demonstrating MLP's superiority in accuracy.

## 🏆 Main Findings
- MLP achieved the highest accuracy (~52.4%)
- Decision Tree had the best F1-score
- Wilcoxon Test confirms significant differences between models
- Performance is moderate due to:
  - Class imbalance
  - Limited dataset size
  - Complexity of health-related behaviors

## 🧰 Requirements
Create a `requirements.txt` file:

```txt
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
jupyterlab
scipy
````

## 📈 Visual Outputs

AUSMI.ipynb includes:
- Heatmaps
- Feature distribution plots
- Confusion matrices
- Wilcoxon test matrices

## 🔭 Future Work
- Increase dataset size
- Apply SMOTE or cost-sensitive learning
- Explore stacking and deep learning models
- Add SHAP/LIME explainability
- Validate models on external datasets

## 📚 References
- United Nations — World Population Prospects 2019
- National Poll on Healthy Aging (NPHA)
- UCI Machine Learning Repository
- Kaggle NPHA projects
