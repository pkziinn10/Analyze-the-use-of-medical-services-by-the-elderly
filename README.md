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
- 2+ visits (Merged the original 2-3 and 4+ classes for a more balanced binary task)

## ⚙️ Methodology Overview
- **Data Cleaning & Feature Selection:** Removal of redundant or inconsistent features, plus dropping the bottom 12 low-importance features using MDI (Mean Decrease Impurity) consensus.
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

The models were evaluated using 10-fold stratified cross-validation and results showed that **XGBoost and Random Forest** achieved the highest accuracy, while **KNN** obtained the best F1-score.

## 📊 Results
**Mean ± Standard Deviation across 10-fold CV**

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| XGBoost | 0.8166 | 0.7348 |
| Random Forest | 0.8166 | 0.7348 |
| MLP Classifier | 0.8110 | 0.7320 |
| Logistic Regression | 0.8110 | 0.7320 |
| KNN | 0.8095 | 0.7398 |
| Decision Tree | 0.8067 | 0.7322 |
| SVM | 0.7801 | 0.7291 |

- **Cross-Validation:** Each model was trained and evaluated on 10 folds, ensuring robust performance.
- **Statistical Test:** The Wilcoxon test was applied to compare distributions and select the best models with statistical confidence.

## 🏆 Main Findings
- Merging the overlapping targets into a binary problem ("0-1 visits" vs "2+ visits") dramatically improved predictability.
- Removing the bottom 12 noise features using MDI (Mean Decrease Impurity) helped clarify decision boundaries.
- **XGBoost** and **Random Forest** achieved the highest accuracy (~81.66%).
- **KNN** achieved the best F1-score (~73.98%).

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
