import pandas as pd
from math import log2

df = pd.read_csv('./Data/Tennis.csv')

def entropy(vals):
    value_counts = vals.value_counts(normalize=True)
    return -sum(p * log2(p) for p in value_counts if p > 0)

def info_gain(df, attr, target):
    return entropy(df[target]) - sum(
        (len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target]) for v in df[attr].unique()
    )

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

attrs = list(df.columns[:-1])
tree = id3(df, 'Play Tennis', attrs)
print("Decision Tree:", tree)
