from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# Load dataset
data = datasets.load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test the classifier
accuracy = knn.score(X_test, y_test)

# Print accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')

# Make a prediction
sample = [X_test[0]]  # First test sample
prediction = knn.predict(sample)
print(f'Predicted Label for Sample: {prediction[0]}')