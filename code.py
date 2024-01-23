import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/content/diabetes.csv')

data.head()

data.describe()

data.info()

"""Checking Null Values"""

data.isna().sum()

"""Checking Duplicated Values"""

data.duplicated().sum()

"""**Data Visualisation**"""

plt.figure(figsize=(12,6))
sns.countplot(x='Outcome', data = data)
plt.show()

"""Observing Outliers"""

plt.figure(figsize=(12,12))
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']):
  plt.subplot(3,3, i+1)
  sns.boxplot(x=col,data = data)
plt.show()

sns.pairplot(data, hue = 'Outcome')
plt.show()

plt.figure(figsize = (12,12))
for i, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']):
  plt.subplot(3,3, i+1)
  sns.histplot(x=col,data = data, kde=True)
plt.show()

plt.figure(figsize=(12,12))
sns.heatmap(data.corr(), vmin=-1.0, center=0, cmap='RdBu_r', annot=True)
plt.show()

"""**Standard Scaling and Label Encoding**"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"],axis=1),),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

X.head()

y = data['Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):
  knn = KNeighborsClassifier(i)
  knn.fit(X_train,y_train)

  train_scores.append(knn.score(X_train,y_train))
  test_scores.append(knn.score(X_test,y_test))

max_train_score = max(train_scores)
train_scores_index=[i for i, v in enumerate(train_scores) if v==max_train_score]
print("Max Train Score {} % and k = {}".format(max_train_score*100, list(map(lambda x:x+1, train_scores_index))))

max_test_score = max(test_scores)
test_scores_index=[i for i, v in enumerate(test_scores) if v==max_test_score]
print("Max Test Score {} % and k = {}".format(max_test_score*100, list(map(lambda x:x+1, test_scores_index))))

plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15), train_scores, marker = '*', label ='Train Scores')
p = sns.lineplot(range(1,15), test_scores, marker = 'o', label ='Test Scores')

knn = KNeighborsClassifier(13)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
