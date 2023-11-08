import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
digits = datasets.load_digits()

# Flatten the images, to turn data in a (samples, feature) matrix
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

# Print the classification report
report = metrics.classification_report(y_test, predicted)
print(report)
