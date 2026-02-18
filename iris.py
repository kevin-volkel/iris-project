
#? ---------------------------------
#? AUTHOR: Kevin Volkel
#? PROJECT: Iris
#? ---------------------------------

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

iris = load_iris()

#? Features
X = iris.data # type: ignore

#? Target
y = iris.target # type: ignore


#? Feature Names
feature_names = iris.feature_names # type: ignore

#? Target Names
target_names = iris.target_names # type: ignore

print("Features: ", feature_names)
print("Target Names: ", target_names)
print("First 5 samples: ", X[:5])

#! Data Preprocessing Step 1: Check for missing, incomplete, or duplicate data
data = pd.DataFrame(X, columns=feature_names)

#? Check for missing values
print("\nMissing values per feature: ", data.isnull().sum())

#? Check for duplicates
print("Number of duplicate rows: ", data.duplicated().sum())

#? Drop the duplicates
data.drop_duplicates(inplace=True)

#? Post dropping
print("\nAfter dropping duplicates:\nMissing values per feature: ", data.isnull().sum())
print("Number of duplicate rows: ", data.duplicated().sum())

#! Data Preprocessing Step 2: Standarize Features Values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

#? Fit the scaler to the training data and transform both training and training sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("First 5 rows of scaled training data: ", X_train_scaled[:5])

#! Step 3: Train and Evaluate the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


#? Save the trained model and scaler to a file
with open("model_and_scaler.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)
    
print("\nModel and scaler has been saved to: 'model_and_scaler.pkl'.")

#? Make predictions on the test data
y_pred = model.predict(X_test_scaled)

#? Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

#? Generate a classification report
print("Classification Report: \n", classification_report(y_test, y_pred, target_names=target_names))

#? Conf Matrix are used to evalutate performance of classification models
#? They break down predictions compared to actual outcomes.
cm = confusion_matrix(y_test, y_pred)

#? Visual the C-matrix in a heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Iris Project")
plt.show()