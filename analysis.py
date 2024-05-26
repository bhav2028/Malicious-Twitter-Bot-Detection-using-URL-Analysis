import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv('bot_detection_data.csv')

# Data cleaning
print(df.shape)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Drop irrelevant columns for EDA and modeling
df_cleaned = df.drop(['User ID', 'Username', 'Tweet', 'Location', 'Created At', 'Hashtags', 'Post url', 'Tweet Link'], axis=1)

# Fill missing values (if any) or handle them based on your data
df_cleaned['Follower Count'].fillna(df_cleaned['Follower Count'].mean(), inplace=True)

# EDA
# Summary statistics
print("Summary Statistics:\n", df_cleaned.describe())

# Correlation matrix
corr_matrix = df_cleaned.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlational Matrix")
plt.show()

# Pairplot for selected feautres
sns.pairplot(df_cleaned[['Retweet Count', 'Mention Count', 'Follower Count', 'Sentiment', 'Bot Label']], hue='Bot Label', markers=["o", "s"])
plt.title("Pairplot of Selected Features")
plt.show()

# Data Preprocessing

print(df_cleaned.columns)


# Split the data into features (X) and target variable (y)
X = df_cleaned.drop('Bot Label', axis=1)
y = df_cleaned['Bot Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()


# Saving the model
joblib.dump(rf_model, 'bot_detection_model.pkl')

# Prediction
y_pred = rf_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
