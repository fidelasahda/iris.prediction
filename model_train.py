from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load the Iris dataset
X, y = load_iris(return_X_y=True)
iris = load_iris()
target_names = iris.target_names 
# Step 2: Train-test split (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Save model to file
joblib.dump(model, "model_train.pkl")

print("✅ Model saved as model_train.pkl")
