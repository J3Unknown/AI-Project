import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Cache the plot generation separately with persist=True
@st.cache_data(persist=True)
def generate_plots(models, X_test_scaled, ytest, df, dt, features):
    # Accuracy comparison plot
    accuracies = {name: accuracy_score(ytest, model.predict(X_test_scaled)) 
                 for name, model in models.items()}
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(accuracies.keys(), accuracies.values(), 
                  color=["skyblue", "lightgreen", "lightcoral", "darkcyan", "lightgray"])
    ax1.set_ylim(0.65, 0.90)
    ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylabel("Accuracy")
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # Correlation matrix
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title("Feature Correlation Matrix")

    # Decision tree visualization
    fig3, ax3 = plt.subplots(figsize=(20, 10))
    plot_tree(dt, feature_names=features, class_names=["0", "1"], filled=True, ax=ax3)

    # Confusion matrices
    fig4, axs = plt.subplots(2, 2, figsize=(14, 12))
    for idx, (name, model) in enumerate(models.items()):
        if name == 'Voting Classifier': 
            continue
        cm = confusion_matrix(ytest, model.predict(X_test_scaled))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[idx // 2, idx % 2])
        axs[idx // 2, idx % 2].set_title(name)

    return {
        'accuracy': fig1,
        'correlation': fig2,
        'tree': fig3,
        'confusion': fig4,
        'accuracies': accuracies
    }

@st.cache_data
def load_and_train():
    # Load and preprocess data
    df = pd.read_csv("heart.csv")
    df["chol"] = pd.to_numeric(df["chol"], errors="coerce")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Split data
    x = df.drop("target", axis=1)
    y = df["target"]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Feature selection
    numeric_cols = xtrain.select_dtypes(include=["number"]).columns
    z_scores = xtrain[numeric_cols].apply(lambda x: np.abs((x - x.mean()) / x.std()))
    mean_z_scores = z_scores.mean().sort_values(ascending=False)

    n_features = len(mean_z_scores)
    low_priority = mean_z_scores[2 * (n_features // 3):].index.tolist()
    xtrain_selected = xtrain.drop(low_priority, axis=1)
    xtest_selected = xtest.drop(low_priority, axis=1)
    features = xtrain_selected.columns.tolist()

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(xtrain_selected)
    X_test_scaled = scaler.transform(xtest_selected)

    # Train individual models
    svm = SVC(kernel="poly", probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression()
    dt = DecisionTreeClassifier(random_state=42, max_depth=6)

    models = {
        'SVM': svm.fit(X_train_scaled, ytrain),
        'KNN': knn.fit(X_train_scaled, ytrain),
        'Logistic Regression': lr.fit(X_train_scaled, ytrain),
        'Decision Tree': dt.fit(X_train_scaled, ytrain)
    }

    # Create voting classifier
    voting = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('knn', knn),
            ('lr', lr),
            ('dt', dt)
        ],
        voting='soft'
    )
    voting.fit(X_train_scaled, ytrain)
    models['Voting Classifier'] = voting

    # Generate plots (now using the separate cached function)
    plot_data = generate_plots(models, X_test_scaled, ytest, df, dt, features)

    return {
        'models': models,
        'scaler': scaler,
        'features': features,
        'plots': {
            'accuracy': plot_data['accuracy'],
            'correlation': plot_data['correlation'],
            'tree': plot_data['tree'],
            'confusion': plot_data['confusion']
        },
        'accuracies': plot_data['accuracies']
    }

# Load models and artifacts
artifacts = load_and_train()

# Streamlit GUI
st.title("Heart Disease Prediction")
st.sidebar.header("Patient Information")

# Input mappings
cp_options = {
    'No Pain': 0,
    'Low Pain': 1,
    'Medium Pain': 2,
    'Hard Pain': 3
}

thal_options = {
    'Normal': 1,
    'Fixed Defect': 2,
    'Reversible Defect': 3
}

# Create input dictionary
input_data = {}
for feature in artifacts['features']:
    match feature:
        case 'age':
            input_data[feature] = st.sidebar.number_input('Age', 0, 150, 21)
        case 'sex':
            input_data[feature] = 1 if st.sidebar.selectbox('Sex', ['Male', 'Female']) == 'Male' else 0
        case 'cp':
            input_data[feature] = cp_options[st.sidebar.selectbox('Chest Pain Type', list(cp_options.keys()))]
        case 'trestbps':
            input_data[feature] = st.sidebar.number_input('Resting BP (mmHg)', 50, 250, 120)
        case 'chol':
            input_data[feature] = st.sidebar.number_input('Cholesterol (mg/dl)', 50, 700, 200)
        case 'fbs':
            input_data[feature] = st.sidebar.selectbox('Fasting Blood Sugar > 120mg/dl', [0, 1])
        case 'restecg':
            input_data[feature] = st.sidebar.selectbox('Resting ECG', [0, 1, 2])
        case 'thalach':
            input_data[feature] = st.sidebar.number_input('Max Heart Rate', 60, 220, 150)
        case 'exang':
            input_data[feature] = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
        case 'oldpeak':
            input_data[feature] = st.sidebar.number_input('ST Depression', 0.0, 6.2, 0.0)
        case 'slope':
            input_data[feature] = st.sidebar.selectbox('ST Slope', [0, 1, 2])
        case 'ca':
            input_data[feature] = st.sidebar.slider('Major Vessels', 0, 3, 0)
        case 'thal':
            input_data[feature] = thal_options[st.sidebar.selectbox('Thalassemia', list(thal_options.keys()))]

# Create DataFrame and preprocess
input_df = pd.DataFrame([input_data])[artifacts['features']]
input_scaled = artifacts['scaler'].transform(input_df)

# Model selection and prediction
model_name = st.sidebar.selectbox("Select Model", list(artifacts['models'].keys()), 4)
model = artifacts['models'][model_name]

if st.sidebar.button('Predict'):
    prediction = model.predict(input_scaled)[0]
    st.sidebar.subheader("Result")
    if prediction == 1:
        st.sidebar.error("High risk of heart disease")
    else:
        st.sidebar.success("Low risk of heart disease")

# Display plots (these will now remain static)
@st.cache_data
def show_plots():
    st.subheader("Accuracy Comparison")
    st.pyplot(artifacts['plots']['accuracy'])

    st.subheader("Feature Correlation Matrix")
    st.pyplot(artifacts['plots']['correlation'])

    st.subheader("Decision Tree Visualization")
    st.pyplot(artifacts['plots']['tree'])

    st.subheader("Confusion Matrices")
    st.pyplot(artifacts['plots']['confusion'])

show_plots()