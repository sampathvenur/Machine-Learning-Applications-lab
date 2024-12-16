from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data
data = {
    'text': [
        'I love programming in Python', 'Python is an amazing language',
        'I hate getting errors in my code', 'Debugging can be frustrating',
        'Machine learning is fascinating', 'I dislike syntax errors'
    ],
    'label': ['positive', 'positive', 'negative', 'negative', 'positive', 'negative']
}

# Vectorize text and split data
X = CountVectorizer().fit_transform(data['text'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
clf = MultinomialNB().fit(X_train, y_train)

# Predictions and accuracy
y_pred = clf.predict(X_test)
print(f"Accuracy of the classifier: {accuracy_score(y_test, y_pred)}")

# Sample prediction
sample = ['I love coding in Python']
sample_vec = CountVectorizer().fit(data['text']).transform(sample)
print(f"Predicted label for the sample text: {clf.predict(sample_vec)[0]}")
