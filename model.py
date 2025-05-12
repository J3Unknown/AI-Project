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


df = pd.read_csv('heart.csv') #create the data frame and read data from csv file
# Data Cleansing
df['chol'] = pd.to_numeric(df['chol'],errors='coerce') #convert non numerical values to NaN to be dropped
df.dropna(inplace=True,how='any')

df.info()
df.drop_duplicates(inplace=True) #drop any duplicates

#print the description after removing the duplicates and transposes the matrix to show all columns
df.describe(include='all')
#calculate the IQR to remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)

df.describe(include='all')
# Feature Selection
# Calculate Z-scores for all numerical columns
numeric_cols = df.select_dtypes(include=['number']).columns
numeric_cols = [col for col in numeric_cols if col != 'target']  # Exclude 'target'
z_scores = df[numeric_cols].apply(lambda x: np.abs((x - x.mean()) / x.std()))
mean_z_scores = z_scores.mean().sort_values(ascending=False)

# Display Z-scores for each column
print("Z-scores for each column:")
print(mean_z_scores)
n_features = len(mean_z_scores)
high_priority = mean_z_scores[:n_features//3].index.tolist()
medium_priority = mean_z_scores[n_features//3 : 2*(n_features//3)].index.tolist()
low_priority = mean_z_scores[2*(n_features//3):].index.tolist()

print(high_priority)
print(medium_priority)
print(low_priority)
low_priority_to_drop = [col for col in low_priority if col in df.columns]
df.drop(low_priority_to_drop, axis=1, inplace=True)

features = high_priority + medium_priority
print(features)
#Extract the X and Y values for the target
xTarget = df.drop('target',axis=1)
yTarget = df['target']

#split the data into train data and test data
xtrain, xtest, ytrain, ytest = train_test_split(xTarget, yTarget, test_size=0.2, random_state=42)

xtrain_selected = xtrain[features]
xtest_selected = xtest[features]