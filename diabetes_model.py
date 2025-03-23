""" Phase 1 - Model Training and Saving """

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset 
df = pd.read_csv("D:\Coding Journey\Web_Dev\proto_medic\diabetes.csv")

# Define featurres and target variables 
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistics Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler using pickle
with open("diabetes_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully")




""" Phase 2 - Model Evaluation """

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Get predictions
y_pred = model.predict(X_test)

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# lassification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()




""" Phase 3 - Prediction """

import pickle
import numpy as np

# This is to supress the StandardScaler warning when loading the model
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the saved model
with open("diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Take user input
def predict_diabetes():
    print("\n🔹 Enter patient details to predict diabetes risk:")
    features = []
    feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]

    for name in feature_names:
        value = float(input(f"Enter {name}: "))
        features.append(value)

    # Convert input to numpy array & scale
    input_data = np.array([features]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    print("\n💡 Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")

predict_diabetes()
