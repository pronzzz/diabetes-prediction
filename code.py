import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Read the data
data = pd.read_csv('diabetes.csv')

# Print the first few rows of the data
print(data.head())

# Print the summary statistics of the data
print(data.describe())

# Print the data types of each feature
print(data.info())

# Check for null values
print(data.isna().sum())

# Check for duplicated values
print(data.duplicated().sum())

# Plot the distribution of the Outcome variable
plt.figure(figsize=(12,6))
sns.countplot(x='Outcome', data = data)
plt.show()

# Plot boxplots of each feature
plt.figure(figsize=(12,12))
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']):
    plt.subplot(3,3, i+1)
    sns.boxplot(x=col,data = data)
plt.show()

# Plot a pairplot of the data
sns.pairplot(data, hue = 'Outcome')
plt.show()

# Plot histograms of each feature
plt.figure(figsize = (12,12))
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']):
    plt.subplot(3,3, i+1)
    sns.histplot(x=col,data = data, kde=True)
plt.show()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
plt.show()

# Split the data into training and test sets
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardize the training and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier with different values of k
test_scores = []
train_scores = []
for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

# Find the optimal value of k
max_train_score = max(train_scores)
train_scores_index=[i for i, v in enumerate(train_scores) if v==max_train_score]
print("Max Train Score {} % and k = {}".format(max_train_score*100, list(map(lambda x:x+1, train_scores_index))))

max_test_score = max(test_scores)
test_scores_index=[i for i, v in enumerate(test_scores) if v==max_test_score]
print("Max Test Score {} % and k = {}".format(max_test_score*100, list(map(lambda x:x+1, test_scores_index))))

# Plot the training and test scores for different values of k
plt.figure(figsize=(12,5))
sns.lineplot(x=range(1,15), y=test_scores, marker = 'o', label ='Test Scores')
sns.lineplot(x=range(1,15), y=train_scores, marker = '*', label ='Train Scores')
plt.show()

# Train the KNN classifier with the optimal value of k
knn = KNeighborsClassifier(13)
knn.fit(X_train, y_train)

# Evaluate the classifier on the test set
score = knn.score(X_test, y_test)
print("Accuracy:", score)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Print the confusion matrix
confusion_matrix(y_test,y_pred)

# Print the classification report
print(classification_report(y_test,y_pred))﻿
