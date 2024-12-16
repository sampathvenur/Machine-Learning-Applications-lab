import numpy as np, csv

data = np.array(list(csv.reader(open('./Data/enjoysports.csv'))))
concepts, target = data[:, :-1], data[:, -1]

s_h = list(concepts[0])
g_h = [['?' for _ in s_h] for _ in s_h]

for i, val in enumerate(concepts):
    if target[i] == "yes":
        s_h = ['?' if s_h[x] != val[x] else s_h[x] for x in range(len(s_h))]
        g_h = [g for g in g_h if not any(g[x] != '?' and g[x] != val[x] for x in range(len(s_h)))]
    else:
        for x in range(len(s_h)):
            if s_h[x] != val[x]:
                g_h[x][x] = s_h[x]

g_h = [g for g in g_h if g != ['?' for _ in s_h]]

print("Concepts:\n", concepts)
print("Target:\n", target)
print("Final Specific_h:\n", np.array(s_h))
print("Final General_h:\n", [list(map(str, h)) for h in g_h])