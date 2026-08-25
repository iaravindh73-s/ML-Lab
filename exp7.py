import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Upload dataset
from google.colab import files
upload = files.upload()

# Read dataset
df = pd.read_csv('heartdisease.csv')

# Explore dataset
print(df.head())
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nColumns:")
print(df.columns.tolist())

# Install pgmpy
!pip install pgmpy==1.0.0

# Import Bayesian Network libraries
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# Build Bayesian Network
model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('Gender', 'heartdisease'),
    ('Family', 'heartdisease'),
    ('diet', 'heartdisease'),
    ('Lifestyle', 'heartdisease'),
    ('cholestrol', 'heartdisease')
])

# Maximum Likelihood Estimation
mle = MaximumLikelihoodEstimator(model, df)

# Calculate CPDs
cpds = mle.get_parameters()

# Add CPDs to model
model.add_cpds(*cpds)

# Display CPDs
print("\nHeart Disease CPD:")
print(model.get_cpds('heartdisease'))

print("\nAge CPD:")
print(model.get_cpds('age'))

print("\nGender CPD:")
print(model.get_cpds('Gender'))

print("\nFamily CPD:")
print(model.get_cpds('Family'))