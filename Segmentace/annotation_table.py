import pandas as pd
import os
from collections import Counter

folder = r"Y:\Datasets\Fyzio"
subfolder = "exercises"
exercises = []
annotations = []
rows = []


for date in os.listdir(folder):
    date_path = os.path.join(folder, date)
    if not os.path.isdir(date_path):
        continue
    for pacient in os.listdir(date_path):
        pacient_path = os.path.join(date_path, pacient)
        if os.path.isdir(pacient_path):
            path = os.path.join(pacient_path, subfolder)
            try:
                for exercise in os.listdir(path):
                    df = pd.read_csv(os.path.join(path, exercise))
                    exercises.append(exercise.strip(".csv"))
                    annotations.append(df.iloc[0].to_dict())
            except:
                pass


exercise_indices = {}
exercise_count = Counter(exercises)
unique_exercises = list(set(exercises))

for exercise in unique_exercises:
    indices = [i for i, ex in enumerate(exercises) if ex == exercise]
    exercise_indices[exercise] = indices


for key, mask in exercise_indices.items():
    tuple_dict = {key: (0, 0)
                  for key in annotations[mask[0]].keys()
                  if key != 'overall quality'}
    for indices in exercise_indices[key]:
        for k, value in annotations[indices].items():
            if k != 'overall quality':
                true_count, false_count = tuple_dict[k]
                if value == 1:
                    true_count += 1
                else:
                    false_count += 1
                tuple_dict[k] = (true_count, false_count)


    for metric, (yes, no) in tuple_dict.items():
        rows.append({
            "Exercise": key,
            "Metric": metric,
            "Yes": yes,
            "No": no
        })

df = pd.DataFrame(rows)
df.to_excel("exercise_summary.xlsx", index=False)
print("Soubor 'exercise_summary.xlsx' byl vytvořen.")