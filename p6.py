import pandas as pd
import math

data = pd.read_csv('./Data/Heart_Disease.csv')

P_A = data['CHDRisk'].value_counts(normalize=True).to_dict()
print("P(A) :", P_A)

P_X_A = {col: data.groupby([col, 'CHDRisk']).size().unstack().apply(lambda x: x/x.sum(), axis=1).to_dict() for col in data.columns[:-1]}
print("P(X/A) :", P_X_A)

P_X = {col: data[col].value_counts(normalize=True).to_dict() for col in data.columns[:-1]}
print("P(X) :", P_X)

def naive_bayes_predict(row):
    probs = {c: math.log(P_A[c]) for c in P_A}
    
    for col in data.columns[:-1]:
        for val in P_X_A[col]:
            if row[col] == val:
                for c in P_A:
                    probs[c] += math.log(P_X_A[col].get(val, {}).get(c, 1e-6))

    return max(probs, key=probs.get)

misclassified = 0
total = len(data)
for _, row in data.iterrows():
    predicted = naive_bayes_predict(row)
    if predicted != row['CHDRisk']:
        misclassified += 1

accuracy = (total - misclassified) / total * 100
print(f'Misclassification Count={misclassified}')
print(f'Misclassification Rate={misclassified/total*100}%')
print(f'Accuracy={accuracy}%')
