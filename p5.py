import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data
data = {
    'test': [
        'I love programming in python',
        'Python is an amazing language',
        'I hate getting errors in my code',
        'Debugging can be frustrating',
        'Machine learning is fascinating',
        'I dislike syntax errors'
    ],
    'label': [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['test'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# Vectorize and train
vectorizer = CountVectorizer(stop_words='english')
clf = MultinomialNB()
clf.fit(vectorizer.fit_transform(X_train), y_train)

# Test accuracy
accuracy = clf.score(vectorizer.transform(X_test), y_test)
print("Accuracy of the classifier:", accuracy)

# Predict sample
sample_text = ["I enjoy learning about artificial intelligence"]
predicted_label = clf.predict(vectorizer.transform(sample_text))
print("Predicted label for the sample text:", "positive" if predicted_label[0] == 1 else "negative")