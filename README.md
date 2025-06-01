# Clasificación de Cáncer de Mama con Regresión Logística

Este proyecto utiliza el conjunto de datos de cáncer de mama de Wisconsin para construir un modelo de regresión logística que predice si un tumor es maligno o benigno. Se implementa un flujo completo de análisis de datos, desde la preparación hasta la evaluación del modelo, incluyendo balanceo de clases y ajuste de hiperparámetros.

## 📁 Estructura del proyecto

- **Carga y limpieza de datos:** Se renombran las columnas para mayor claridad.
- **Preparación del conjunto de datos:** División en características (X) y etiquetas (y), junto con un split estratificado en entrenamiento y prueba.
- **Balanceo de clases con SMOTE:** Para evitar el sesgo hacia la clase mayoritaria.
- **Ajuste de hiperparámetros con GridSearchCV:** Se busca la mejor combinación de `C`, `penalty` y `solver`.
- **Entrenamiento final y evaluación del modelo:** Se entrena un modelo final usando los mejores parámetros y se evalúa con métricas como AUC, precisión, recall y f1-score.
- **Interpretabilidad:** Se muestran los coeficientes más importantes para comprender la influencia de cada variable.

## 📊 Resultados

- **AUC en conjunto de prueba:** 0.9962
- **Precisión global:** 96%
- Buen equilibrio entre precisión y recall para ambas clases.

## 🧠 Conclusiones

- La regresión logística, combinada con SMOTE y un buen ajuste de hiperparámetros, puede lograr una excelente capacidad predictiva en tareas de clasificación binaria.
- El análisis de coeficientes proporciona una interpretación clara de las variables más relevantes en la predicción.
- A pesar de que la regresión logística asume linealidad, su rendimiento fue sobresaliente, lo que sugiere que los datos tienen una estructura lineal bien separable después del preprocesamiento.

## 📚 Requisitos

- Python 3.7+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- imbalanced-learn (para SMOTE)
- lightgbm (si se desea comparar con modelos de árboles)

Instala los paquetes necesarios con:

```bash
pip install -r requirements.txt
```
# ENGLISH
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
   git clone https://github.com/yourusername/breast-cancer-logistic.git](https://github.com/andresc27/cancer_prediction.git
