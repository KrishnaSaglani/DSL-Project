import numpy as np
import pandas as pd

def noise(val):
    return val + np.random.randint(-20, 20)

data = []

# Morse patterns (ms)
patterns = {
    'A': [100,100,300],
    'B': [300,100,100,100,100],
    'C': [300,100,300,100],
    'D': [300,100,100],
    'E': [100]
}

label_map = {'A':0,'B':1,'C':2,'D':3,'E':4}

for letter, seq in patterns.items():
    for _ in range(200):
        sample = [noise(x) for x in seq]

        # pad to length 5
        while len(sample) < 5:
            sample.append(0)

        data.append(sample + [label_map[letter]])

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)

print("Dataset generated")