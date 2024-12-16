import csv

with open('./Data/enjoysports.csv', 'r') as file:
    reader = csv.reader(file)
    data = [row for row in reader]

hypothesis = ['0'] * (len(data[0]) - 1)

print("The Given Training Data Set")
for row in data:
    print(row)

print("\nThe initial value of hypothesis:")
print(hypothesis)

print("\nFind S: Finding a Maximally Specific Hypothesis")
for i, instance in enumerate(data):
    if instance[-1] == 'yes':
        for j in range(len(hypothesis)):
            if hypothesis[j] == '0':
                hypothesis[j] = instance[j]
            elif hypothesis[j] != instance[j]:
                hypothesis[j] = '?'
        print(f"For Training instance No:{i} the {hypothesis}")

print("\nThe Maximally Specific Hypothesis for a given Training Examples:")
print(hypothesis)