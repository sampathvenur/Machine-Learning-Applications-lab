import pandas as pd
from math import log2

df = pd.read_csv('./Data/enjoysports - Sheet1.csv')

def entropy(vals):
    probs = [vals.count(v) / len(vals) for v in set(vals)]
    return -sum(p * log2(p) for p in probs if p > 0)

def info_gain(df, attr, target):
    vals = df[attr].unique()
    weighted_entropy = sum(
        (len(df[df[attr] == v]) / len(df)) * entropy(list(df[df[attr] == v][target]))
        for v in vals
    )
    return entropy(list(df[target])) - weighted_entropy

def id3(df, target, attrs):
    if len(set(df[target])) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}
    for v in df[best].unique():
        subtree = id3(df[df[best] == v], target, [a for a in attrs if a != best])
        tree[best][v] = subtree
    return tree

attrs = [c for c in df.columns if c != 'Play Tennis']
tree = id3(df, 'Play Tennis', attrs)
print("\nThe Decision Tree is:\n", tree)