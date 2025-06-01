# Breast Cancer Classification using Logistic Regression

This project aims to classify breast cancer tumors as **malignant** or **benign** using a logistic regression model. It includes data preprocessing, class balancing using SMOTE, hyperparameter tuning via GridSearchCV, model training, and evaluation.

## 🧪 Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains 569 observations with 30 numeric features computed from digitized images of breast mass samples.

- Target variable: `Diagnosis` (`M` for malignant, `B` for benign)
- Features: Mean, standard error, and worst values of various cell characteristics (e.g., radius, texture, perimeter, area, etc.)

## 🧰 Tools & Libraries

- Python 3
- pandas
- numpy
- scikit-learn
- imbalanced-learn (for SMOTE)
- matplotlib / seaborn (for visualization)

## ⚙️ Project Structure

- **Data Preprocessing**: Renamed features for clarity, encoded target variable, split into training and test sets using stratified sampling.
- **SMOTE**: Applied to training data to balance class distribution.
- **Model Selection**: Used `GridSearchCV` to tune logistic regression hyperparameters (`C`, `penalty`, `solver`) with ROC AUC as the scoring metric.
- **Model Training**: Trained the best logistic regression model on the balanced training set.
- **Model Evaluation**: Evaluated performance on the test set using AUC, accuracy, precision, recall, and F1-score.
- **Feature Importance**: Interpreted feature coefficients to identify most influential variables.

## 📊 Results

- **AUC Score on Test Set**: 0.996
- **Accuracy**: 96%
- **Recall (Malignant)**: 92%
- **Top Influential Features**: Selected based on absolute value of logistic regression coefficients.

## 📌 Conclusions

- Logistic regression performed excellently for binary classification in this medical context.
- Balancing the data with SMOTE improved the model's ability to detect minority class (malignant cases).
- Model interpretability is retained through feature coefficients.
- Future work may include applying more complex models (e.g., Random Forests, Gradient Boosting) to capture non-linear relationships while preserving interpretability through tools like SHAP or LIME.

## 📁 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-logistic.git
   cd breast-cancer-logistic
