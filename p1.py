import csv

# Read the data
with open('./Data/enjoysports - Sheet1.csv', 'r') as file:
    data = list(csv.reader(file))
    print("\nTraining Data:\n")
    for row in data:
        print(row)

# Initialize hypothesis
hypothesis = data[0][:-1]
print("\nInitial Hypothesis:", hypothesis)

# Find-S Algorithm
print("\nFind-S Algorithm Process:")
for i, row in enumerate(data):
    if row[-1] == 'yes':  # Only consider positive examples
        for j in range(len(hypothesis)):
            if row[j] != hypothesis[j]:
                hypothesis[j] = '?'
    print(f"Hypothesis after instance {i+1}: {hypothesis}")

# Final Hypothesis
print("\nFinal Hypothesis:", hypothesis)