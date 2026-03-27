import random
import pandas as pd

skills = [
    "python", "sql", "dsa", "machine learning", "data analysis",
    "pandas", "numpy", "matplotlib", "power bi", "excel",
    "flask", "api", "statistics"
]

data = []

# 🔥 GOOD MATCHES
for _ in range(250):
    selected = random.sample(skills, 4)
    resume = " ".join(selected)

    job = " ".join(selected[:3])  # overlapping skills
    data.append([resume, job, 1])

# 🔥 BAD MATCHES
for _ in range(250):
    resume = " ".join(random.sample(skills, 4))
    job = "java spring backend developer"
    data.append([resume, job, 0])

df = pd.DataFrame(data, columns=["resume", "job", "label"])
df.to_csv("training_data.csv", index=False)

print("✅ Clean dataset created!")