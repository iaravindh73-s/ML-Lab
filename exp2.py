# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

# Load the spam dataset
from google.colab import files
upload = files.upload()

dataset = pd.read_csv("/content/spam.csv")

# Explore the dataset
dataset.info()
dataset.info

# Splitting our data into X and y.
X = dataset['Category'].values
y = dataset['Message'].values

sns.countplot(x='Category', data=dataset)
plt.savefig("exercise3a.png")
files.download("exercise3a.png")

# Splitting our data into training and testing.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Converting String to Integer
# Converting text into integer using CountVectorizer()
cv = CountVectorizer()

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Applying SVM algorithm
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test, y_test))