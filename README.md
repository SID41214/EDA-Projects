# Exploratory Data Analysis (EDA)

## 📌 Introduction
Exploratory Data Analysis (EDA) is a crucial step in data science and analytics that involves analyzing datasets to summarize their main characteristics. It helps in identifying patterns, spotting anomalies, testing hypotheses, and checking assumptions through statistical summaries and graphical representations.

## 🎯 Objectives of EDA
- Understanding the structure and distribution of data.
- Handling missing and inconsistent values.
- Identifying outliers and anomalies.
- Analyzing correlations between variables.
- Visualizing data trends and patterns for better insights.
- Performing feature engineering and dimensionality reduction.
- Scaling data for model readiness.

## 📦 Installation
To perform EDA efficiently, install the necessary Python libraries:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🔧 Steps in EDA
### 1️⃣ Understanding the Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
print(df.shape)  # Check the number of rows and columns
print(df.head())  # Display first few rows
print(df.info())  # Dataset structure and data types
```

### 2️⃣ Data Cleaning
```python
# Check for missing values
print(df.isnull().sum())

# Handle missing values by filling with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)
```

### 3️⃣ Descriptive Statistics
```python
print(df.describe())  # Summary statistics of numerical columns
```

### 4️⃣ Data Visualization
#### Histogram for Distribution Analysis
```python
plt.figure(figsize=(8,5))
sns.histplot(df['column_name'], bins=30, kde=True)
plt.title("Distribution of Column Name")
plt.xlabel("Column Name")
plt.ylabel("Frequency")
plt.show()
```

#### Scatter Plot for Relationship Analysis
```python
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['feature1'], y=df['feature2'])
plt.title("Feature1 vs Feature2")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()
```

### 5️⃣ Correlation Analysis
```python
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()
```

### 6️⃣ Outlier Detection
```python
plt.figure(figsize=(10,5))
sns.boxplot(data=df)
plt.title("Box Plot for Outlier Detection")
plt.show()
```

### 7️⃣ Feature Engineering
```python
# Creating new feature: total_rooms
if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
```

### 8️⃣ Dimensionality Reduction
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.select_dtypes(include=np.number))
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```

### 9️⃣ Data Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))
```

### 🔟 Initial Modeling
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop(columns=['target_column'])  # Replace with actual target column
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model Score:", model.score(X_test, y_test))
```

## 🤝 Contributing
Contributions to improve this EDA guide are always welcome. Feel free to fork the repository and submit a pull request with your enhancements.


## 📬 Contact
For any queries or contributions, reach out via GitHub or email.
