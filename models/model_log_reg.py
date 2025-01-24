# Step 1: Import libraries & Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('fetal_health.csv')

# EXPLORATORY DATA ANALYSIS
df.info()  # Check the structure
df.describe().T  # Summary statistics
df.isnull().sum()  # Check for missing values
df['fetal_health'].value_counts()  # Check the distribution of the target variable (1 - normal, 2 - suspect, 3 - pathologic)

# Step 4: Clean the data (if needed)
df = df.dropna()  # Drop rows with missing values

# Step 5: Visualization
sns.countplot(x='fetal_health', data=df)  
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')  # Correlation matrix

# DATA PREPROCESSING
from sklearn.preprocessing import StandardScaler

# Scaling data because features are on different scales
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled, columns=df.columns)


# MACHINE LEARNING MODELS for SuperVised Learning - Classification

#Split Data into Training and Testing Sets
from sklearn.model_selection import train_test_split
features = df_scaled.drop('fetal_health', axis=1)
fetal_health_class = df['fetal_health']
features_train, features_test, fetal_health_train, fetal_health_test = train_test_split(features, fetal_health_class, test_size=0.2, random_state=42)

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=50, multi_class='multinomial') #multinomial for multi-class classification when the target variables are mutually exclusive
model.fit(features_train, fetal_health_train)