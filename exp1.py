# Import the required libraries
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn import tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Load Iris dataset
from google.colab import files
upload = files.upload()

# Load data from CSV
iris_df = pd.read_csv("/content/Iris.csv")
iris_df.head()

# Explore the dataset
iris_df.isnull().any()
iris_df.dtypes
iris_df.describe()

# Perform pair plotting to view relationship between the features present in the data
sns.pairplot(iris_df, hue='Species')
plt.savefig("exercise2a.png")
files.download("exercise2a.png")

# Split data into train and test set
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
             'PetalWidthCm']].values
y = iris_df['Species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=1
)

# Train the decision tree classifier
# scikit learn decision tree model training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Test the model trained on test set
prediction = clf.predict(X_test)
prediction

# Perform Evaluation on Test set
# evaluation for multi class classification
print(classification_report(y_test, prediction))

# Plot Confusion Matrix
plot_confusion_matrix(clf, X_test, y_test)
plt.savefig("exercise2b.png")
files.download("exercise2b.png")

# Visualize the Decision Tree
!pip install pydotplus
!apt-get install graphviz -y

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# Visualize the graph
dot_data = StringIO()

export_graphviz(
    clf,
    out_file=dot_data,
    feature_names=['SepalLengthCm', 'SepalWidthCm',
                   'PetalLengthCm', 'PetalWidthCm'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

plt.savefig("exercise2c.png")
files.download("exercise2c.png")

# Feeding new data to the classifier to predict the right class
SepalLengthCm = 4.8
SepalWidthCm = 2.9
PetalLengthCm = 1.3
PetalWidthCm = 0.2

x = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
res = clf.predict(x)

print("The class predicted is --> " + str(*res))