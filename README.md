# Heart Disease Prediction Web App

A Streamlit-based web application for predicting the risk of heart disease using machine learning models.

## Features

- **User-Friendly Interface**: Input patient data via sidebar sliders and dropdowns.
- **Multiple Models**: Choose between SVM, KNN, Logistic Regression, Decision Tree, or a Voting Classifier.
- **Visualizations**:
  - Accuracy comparison across models.
  - Feature correlation matrix.
  - Decision Tree structure.
  - Confusion matrices for individual models.
- **Real-Time Prediction**: Instantly assess heart disease risk (low/high) based on input data.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   _(Create `requirements.txt` with: streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn)_

3. **Run the app**:
   ```bash
   streamlit run model.py
   ```

## Usage

1. Input patient information in the sidebar (e.g., age, cholesterol, blood pressure).
2. Select a model from the dropdown.
3. Click **Predict** to see the risk assessment.
4. Explore visualizations for model performance and data insights.

## Dataset

The `heart.csv` file contains clinical features for heart disease prediction, including:

- `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), and more.
- **Preprocessing**:
  - Missing values and duplicates are removed.
  - Features are scaled using `StandardScaler`.
  - Lower-priority features (bottom 33% by Z-score) are excluded.

## Models

- **Supported Algorithms**:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Decision Tree
  - Voting Classifier (ensemble of all models)
- **Metrics**:
  - Accuracy scores for each model.
  - Confusion matrices to evaluate performance.

## Project Structure

```
.
├── model.py             # Streamlit app and ML model training
├── heart.csv            # Heart disease dataset
├── README.md            # Project documentation
├── AI-Project.ipynb     # Jupyter Notebook file
└── requirements.txt     # Dependency list
```
