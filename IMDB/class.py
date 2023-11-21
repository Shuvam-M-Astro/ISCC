import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time

# Replace this with the path to your converted IMDb dataset CSV
csv_file_path = 'C:/Users/shuva/Downloads/Bodvar/imdb_reviews_train.csv'  # or 'imdb_reviews_test.csv'

# Read the dataset
df = pd.read_csv(csv_file_path)

# Assuming the CSV has columns 'review' and 'sentiment'
X = df['review']  # the reviews
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to a matrix of token counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a simple Naive Bayes classifier
clf = MultinomialNB()
start_time = time.time()
clf.fit(X_train_counts, y_train)
training_duration = time.time() - start_time

# Predict and calculate accuracy
predictions = clf.predict(X_test_counts)
accuracy = accuracy_score(y_test, predictions)

print(f"Training duration: {training_duration} seconds")
print(f"Accuracy: {accuracy * 100}%")
