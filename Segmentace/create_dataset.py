"""
Vytvoří trénovací datasety pro daná cvičení
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


Fs = 200
window_size = Fs
step_size = Fs//2

channels = [
    "Biceps_EMG_Envelope", "Biceps_Q1", "Biceps_Q2", "Biceps_Q3", "Biceps_Q4",
    "Triceps_EMG_Envelope", "Triceps_Q1", "Triceps_Q2", "Triceps_Q3", "Triceps_Q4",
    "Gastrocnemious_EMG_Envelope","Gastrocnemious_Q1","Gastrocnemious_Q2","Gastrocnemious_Q3","Gastrocnemious_Q4",
    "Rectus_EMG_Envelope","Rectus_Q1","Rectus_Q2","Rectus_Q3","Rectus_Q4"
]

data_X = {
    "Burpees": [],
    "High knee run": [],
    "Jump squats": [],
    "Lunge (left leg forward)": [],
    "Lunge (right leg forward)": [],
    "Push-ups": [],
    "Triceps push-ups": [],
    "Wide squat": []
}
data_Y = {key: [] for key in data_X}

path = r"Y:\Datasets\Fyzio\signals_envelope_phase"

for dirpath, dirnames, filenames in os.walk(r"Y:\Datasets\Fyzio"):
    for dir in dirnames:
        if dir == "signals_envelope_phase":
            sub_path = os.path.join(dirpath,dir)
            for file in os.listdir(sub_path):
                file_name = os.path.splitext(file)[0]
                df = pd.read_csv(os.path.join(sub_path,file))
                signal_length = len(df)
                for start in range(0, signal_length - window_size + 1, step_size):
                    end = start + window_size
                    segment_X = df.loc[start:end,channels].to_numpy()
                    segment_Y = df.loc[end-1,["Exercise_Phase"]].values[0]
                    if file_name in data_X:
                        data_X[file_name].append(segment_X)
                        data_Y[file_name].append(segment_Y)

for key, value in data_X.items():
    all_segments = np.array(data_X[key])
    print(all_segments.shape, key)