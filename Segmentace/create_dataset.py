"""
Vytvoří trénovací datasety pro daná cvičení
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

Fs = 200
window_size = 200
step_size = 5

channels = [
    "Biceps_EMG_Envelope", "Biceps_Q1", "Biceps_Q2", "Biceps_Q3", "Biceps_Q4",
    "Triceps_EMG_Envelope", "Triceps_Q1", "Triceps_Q2", "Triceps_Q3", "Triceps_Q4",
    "Gastrocnemious_EMG_Envelope","Gastrocnemious_Q1","Gastrocnemious_Q2","Gastrocnemious_Q3","Gastrocnemious_Q4",
    "Rectus_EMG_Envelope","Rectus_Q1","Rectus_Q2","Rectus_Q3","Rectus_Q4"
]

# channels = [
#      "Biceps_Q1", "Biceps_Q2", "Biceps_Q3", "Biceps_Q4",
#      "Triceps_Q1", "Triceps_Q2", "Triceps_Q3", "Triceps_Q4",
#     "Gastrocnemious_Q1","Gastrocnemious_Q2","Gastrocnemious_Q3","Gastrocnemious_Q4",
#     "Rectus_Q1","Rectus_Q2","Rectus_Q3","Rectus_Q4"
# ]


data_X = {
    # "Burpees": [],
    # "High knee run": [],
    # "Jump squats": [],
    # "Lunge (left leg forward)": [],
    # "Lunge (right leg forward)": [],
    # "Push-ups": [],
    # "Triceps push-ups": [],
    "Wide squat": []
}

validation_X = []
validation_Y = []
data_Y = {key: [] for key in data_X}

path = r"Y:\Datasets\Fyzio\signals_envelope_phase"
save_path = r"Y:\Datasets\Fyzio"
save_dir_X = "X_train_w_transition"
save_dir_Y = "Y_train_w_transition"
fname = "Wide squat"
for dirpath, dirnames, filenames in os.walk(r"Y:\Datasets\Fyzio"):
    for dir in dirnames:
        if dir == "signals_envelope_phase" or dir == "transitions_envelope_phase":
            sub_path = os.path.join(dirpath,dir)
            for file in os.listdir(sub_path):
                file_name = os.path.splitext(file)[0]
                if file_name in data_X or dir == "transitions_envelope_phase":
                    step_size = 25 if dir == "transitions_envelope_phase" else 5
                    df = pd.read_csv(os.path.join(sub_path,file))
                    signal_length = len(df)

                    for start in range(0, signal_length - window_size + 1, step_size):
                        end = start + window_size
                        segment_X = df.loc[start:end-1,channels].to_numpy()
                        segment_Y = df.loc[end-1,["Exercise_Phase"]]
                        if np.isnan(segment_X).any():
                            print(file_name,start,end)
                        else:
                            data_X[fname].append(segment_X)
                            data_Y[fname].append(segment_Y)

os.makedirs(os.path.join(save_path,save_dir_X),exist_ok=True)
os.makedirs(os.path.join(save_path,save_dir_Y),exist_ok=True)

x = np.array(data_X[fname].copy())
x = x.reshape(-1,20)

df = pd.DataFrame(x)
df.to_csv("X.csv",index=False)
df = pd.DataFrame(data_Y["Wide squat"])
df.to_csv("Y.csv",index=False)

for X_key, _ in data_X.items():
    with open(os.path.join(save_path,save_dir_X,X_key+".pkl"),"wb") as f:
        pickle.dump(data_X[X_key],f)

for Y_key, value in data_Y.items():
    with open(os.path.join(save_path, save_dir_Y, Y_key + ".pkl"), "wb") as f:
        pickle.dump(data_Y[Y_key], f)
