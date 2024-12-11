import pandas as pd

# Calculate probabilities
def train(data, target):
    classes = data[target].unique()
    probs = {cls: len(data[data[target] == cls]) / len(data) for cls in classes}
    attr_probs = {
        col: {
            val: {
                cls: len(data[(data[col] == val) & (data[target] == cls)]) / len(data[data[target] == cls])
                for cls in classes
            }
            for val in data[col].unique()
        }
        for col in data.columns[:-1]
    }
    return probs, attr_probs

# Predict classes
def predict(example, probs, attr_probs):
    predictions = {
        cls: probs[cls] * 
        prod(attr_probs[col][val][cls] for col, val in zip(example.index, example))
        for cls in probs
    }
    return max(predictions, key=predictions.get)

# Main execution
df = pd.read_csv('./Data/Tennis.csv')
class_probs, attr_probs = train(df, 'Play')
example = df.iloc[0, :-1]  # Test with the first row
print("Prediction:", predict(example, class_probs, attr_probs))