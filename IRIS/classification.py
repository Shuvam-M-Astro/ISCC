import time
import subprocess
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def check_gpu():
    try:
        # This command works for NVIDIA GPUs and will throw an error if not available
        subprocess.check_output(["nvidia-smi", "-L"])
        return "NVIDIA GPU detected"
    except subprocess.CalledProcessError as e:
        return "NVIDIA GPU not detected"
    except FileNotFoundError as e:
        return "nvidia-smi not found, cannot detect NVIDIA GPU"

# Print the execution device information
print(check_gpu())

# Start the timer
start_time = time.time()

# Load the IRIS dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
clf = LogisticRegression(random_state=42, max_iter=200)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

end_time = time.time()

print(f"Accuracy: {accuracy}")
print(report)
print(f"Time taken: {end_time - start_time:.2f} seconds")
