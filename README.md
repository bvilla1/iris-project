# 🌸 Iris Species Classification

**Final Project — Data Mining | Universidad de la Costa**

## Description
Interactive dashboard that classifies Iris flower species using a Random Forest model trained on the classic Iris dataset.

## Team Members
- (BREINER-VILLA)

## How to Run

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run Proyect.py
```

## Features
- **Model Metrics**: Accuracy, Precision, Recall, F1 Score
- **Interactive Prediction**: Enter sepal/petal measurements and get instant species prediction
- **3D Scatter Plot**: Visualize your sample vs the full dataset
- **Data Exploration**: Histograms, scatter matrix, and boxplots

## Dataset
The [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) contains 150 samples of 3 Iris species with 4 features each.

## Methodology
1. Data loading and understanding
2. Train/test split (80/20)
3. Feature scaling (StandardScaler)
4. Random Forest Classifier (100 trees)
5. Evaluation with multiple metrics
