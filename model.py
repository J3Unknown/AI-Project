import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("heart.csv")  # create the data frame and read data from csv file
# Data Cleansing
df["chol"] = pd.to_numeric(
    df["chol"], errors="coerce"
)  # convert non numerical values to NaN to be dropped
df.dropna(inplace=True, how="any")

df.info()
df.drop_duplicates(inplace=True)  # drop any duplicates

# print the description after removing the duplicates and transposes the matrix to show all columns
df.describe(include="all")
# Extract the X and Y values for the target
xTarget = df.drop("target", axis=1)
yTarget = df["target"]

# split the data into train data and test data
xtrain, xtest, ytrain, ytest = train_test_split(
    xTarget, yTarget, test_size=0.2, random_state=42
)
# Remove missing values and duplicates from TRAINING data
xtrain_clean = xtrain.dropna(how="any")
xtrain_clean = xtrain_clean.drop_duplicates()

xtest_clean = xtest.dropna(how="any")
xtest_clean = xtest_clean.drop_duplicates()
# calculate the IQR to remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# Clip outliers in TRAINING data
xtest_clean = xtest_clean.clip(lower=lower_bound, upper=upper_bound, axis=1)


df.describe(include="all")
# Feature Selection
# Calculate Z-scores for all numerical columns
numeric_cols = xtrain_clean.select_dtypes(include=["number"]).columns
numeric_cols = [col for col in numeric_cols if col != "target"]  # Exclude 'target'
z_scores = xtrain_clean[numeric_cols].apply(lambda x: np.abs((x - x.mean()) / x.std()))
mean_z_scores = z_scores.mean().sort_values(ascending=False)

# Display Z-scores for each column
print("Z-scores for each column:")
print(mean_z_scores)
n_features = len(mean_z_scores)
high_priority = mean_z_scores[: n_features // 3].index.tolist()
medium_priority = mean_z_scores[n_features // 3 : 2 * (n_features // 3)].index.tolist()
low_priority = mean_z_scores[2 * (n_features // 3) :].index.tolist()

print(high_priority)
print(medium_priority)
print(low_priority)
low_priority_to_drop = [col for col in low_priority if col in xtrain_clean.columns]
xtrain_selected = xtrain_clean.drop(low_priority_to_drop, axis=1)
xtest_selected = xtest_clean.drop(low_priority_to_drop, axis=1)

features = high_priority + medium_priority
print(features)
#                      Scalling Data
# converts the data into ranges between 0 and 1 used for KNN and SVM
minMaxScaler = MinMaxScaler()

# fitting the High Priority Data
xTrain_highmd = minMaxScaler.fit_transform(xtrain_selected)
xTest_highmd = minMaxScaler.transform(xtest_selected)

# converting the fitted data into data frames to be easier for tracking
xTrain_df = pd.DataFrame(xTrain_highmd, columns=features, index=xtrain.index)
xTest_df = pd.DataFrame(xTest_highmd, columns=features, index=xtest.index)

# to convert data into Z-Score values to be used in logistic regression and decision tree
scaler = StandardScaler()

# fitting the High Priority Data
X_train_highmed = scaler.fit_transform(xtrain_selected)
X_test_highmed = scaler.transform(xtest_selected)

# recreating the data into data frames to be easier for tracking
x_train_scaled = pd.DataFrame(X_train_highmed, columns=features, index=xtrain.index)
X_test_scaled = pd.DataFrame(X_test_highmed, columns=features, index=xtest.index)
# Model Training

## svm model creation
SVM_model = SVC(kernel="poly")
SVM_model.fit(x_train_scaled, ytrain)

svm_score = SVM_model.score(X_test_scaled, ytest)
svm_pre = SVM_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(ytest, svm_pre)
svm_CMatrix = confusion_matrix(ytest, svm_pre)
svm_classification_report = classification_report(ytest, svm_pre)

### print section
print("\nSVM Result: ")
print(f"SVM Score : {svm_score}")
print(f"SVM Accuracy : {svm_accuracy}")
print(f"SVM Confusion Matrix :\n {svm_CMatrix}")
print(f"SVM Classification Report :\n {svm_classification_report}")


## KNN model creation
KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(x_train_scaled, ytrain)

knn_score = KNN_model.score(X_test_scaled, ytest)
knn_pre = KNN_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(ytest, knn_pre)
knn_CMatrix = confusion_matrix(ytest, knn_pre)
knn_classification_report = classification_report(ytest, knn_pre)

### print section
print("\nKNN Result: ")
print(f"KNN Score : {knn_score}")
print(f"KNN Accuracy : {knn_accuracy}")
print(f"KNN Confusion Matrix :\n {knn_CMatrix}")
print(f"KNN Classification Report :\n {knn_classification_report}")

logreg_accuracy = 0.8  # Will change after make the models
dt_accuracy = 0.85  # Will change after make the models
model_names = ["SVM", "KNN", "Logistic Regression", "Decision Tree"]
accuracies = [svm_accuracy, knn_accuracy, logreg_accuracy, dt_accuracy]

# Make the figure to show each model accuracy
plt.figure(figsize=(10, 6))
bars = plt.bar(
    model_names, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
)

plt.ylabel("Accuracy", fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylim(0.65, 0.90)

# Add values of accuracies on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Correlation Matrix
correlation_matrix = df.corr()

# Make the figure to show the heat map between the features and the target
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"label": "Correlation Coefficient"},
    linewidths=0.5,
    annot_kws={"size": 10},
)

plt.title("Feature Correlation Matrix with Target", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Make the figure to show confusion matrices for the models
plt.figure(figsize=(14, 12))
plt.suptitle("Confusion Matrices Comparison", y=1.02, fontsize=16)

models = [("SVM", svm_CMatrix), ("KNN", knn_CMatrix)]

for idx, (model_name, matrix) in enumerate(models, 1):
    plt.subplot(2, 2, idx)
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14}
    )
    plt.title(f"{model_name}", fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks([0.5, 1.5], ["No Heart Disease", "Heart Disease"], rotation=0)
    plt.yticks([0.5, 1.5], ["No Heart Disease", "Heart Disease"], rotation=0)

plt.tight_layout()
plt.show()
