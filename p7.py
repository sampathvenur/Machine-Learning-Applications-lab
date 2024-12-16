import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./Data/Heart_Disease.csv')

data_encoded = pd.get_dummies(data.drop(columns=['CHDRisk']), drop_first=True)

X = data_encoded
y = data['CHDRisk'].map({'no': 0, 'yes': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total number of Training Data: {X_train.shape}")
print(f"Total number of Test Data: {X_test.shape}")

model = GaussianNB()
model.fit(X_train, y_train)

predicted = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predicted)
print(f"Accuracy of the classifier: {accuracy}")

conf_matrix = metrics.confusion_matrix(y_test, predicted)

print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(f"Predicted Value for individual Test Data: {predicted[0]}")