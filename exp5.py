# Install libraries
!mamba install scikit-learn
!mamba install pandas
!pip install seaborn

# Import functions
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from sklearn import datasets

# Importing dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

print(iris_data)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    iris_data,
    iris_labels,
    test_size=0.20,
    random_state=42
)

# Train model
classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Running Predictions
print("accuracy is")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Checking Validation
import matplotlib.pyplot as plt
import seaborn as sn

plt.figure(figsize=(10, 10))
sn.heatmap(cm, annot=True)

plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.savefig("exercise6.png")