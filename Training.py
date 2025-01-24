# Step 1: Import libraries & Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from models.models import LogisticRegression

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
features = df_scaled.drop('fetal_health', axis=1)
fetal_health_class = df['fetal_health']
features_train, features_test, fetal_health_train, fetal_health_test = train_test_split(features, fetal_health_class, test_size=0.2, random_state=42)

# TRAINING THE MODEL
def eval(model, features_train, class_train, features_test, class_test):
    class_train_prediction = model.predict(features_train)
    train_accuracy = accuracy_score(class_train, class_train_prediction)
    
    class_test_prediction = model.predict(features_test)
    test_accuracy = accuracy_score(class_test, class_test_prediction)
    
    cm = confusion_matrix(class_test, class_test_prediction)
    
    report = classification_report(class_test, class_test_prediction, target_names=["Class 1", "Class 2", "Class 3"])

    return train_accuracy, test_accuracy, cm, report

def train(model_name: str, model):
    model.fit(features_train, fetal_health_train)
    train_accuracy, test_accuracy, cm, report = eval(model, features_train, fetal_health_train, features_test, fetal_health_test)
    print('model:', model_name, 
          'train accuracy:', train_accuracy, 
          'test accuracy:', test_accuracy)
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(report)


#EVALUATING MODEL
train('Logistic Regression', LogisticRegression())




