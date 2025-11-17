"""
Vytvoří trénovací datasety pro daná cvičení
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Fs = 200
window_size = 200
step_size = 5

channels = [
    "Biceps_EMG_Envelope", "Biceps_Q1", "Biceps_Q2", "Biceps_Q3", "Biceps_Q4",
    "Triceps_EMG_Envelope", "Triceps_Q1", "Triceps_Q2", "Triceps_Q3", "Triceps_Q4",
    "Gastrocnemious_EMG_Envelope","Gastrocnemious_Q1","Gastrocnemious_Q2","Gastrocnemious_Q3","Gastrocnemious_Q4",
    "Rectus_EMG_Envelope","Rectus_Q1","Rectus_Q2","Rectus_Q3","Rectus_Q4"
]

y_channels = ["squat_depth","head_axis","feet"]

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

dataset_type = "Class"


path = r"Y:\Datasets\Fyzio\signals_envelope_phase"
save_path = r"Y:\Datasets\Fyzio"
save_dir_X = f"X_{dataset_type}_train_w_augmentation"
save_dir_Y = f"Y_{dataset_type}_train_w_augmentation"
fname = "Wide squat"

def GetClassY(pathY,exercise):
    try:
        df = pd.read_csv(os.path.join(pathY, exercise + ".csv"))
        seg_Y = df[y_channels]
        print(seg_Y.values[0])
        return seg_Y.values[0]
    except:
        return None

for dirpath, dirnames, filenames in os.walk(r"Y:\Datasets\Fyzio"):
    print(dirpath)
    for dir in dirnames:
        if dir == "signals_envelope_phase" or (dir == "transitions_envelope_phase" and dataset_type == "Reg"):
            sub_path = os.path.join(dirpath,dir)
            for file in os.listdir(sub_path):
                file_name = os.path.splitext(file)[0]
                if file_name in data_X or dir == "transitions_envelope_phase":
                    step_size = 100 if (dir == "transitions_envelope_phase" or dataset_type == "Class") else 5
                    try:
                        df = pd.read_csv(os.path.join(sub_path,file))
                    except Exception as e:
                        # print(e)
                        # print(sub_path,file)
                        continue
                    signal_length = len(df)
                    if dataset_type == "Class" and dir != "transitions_envelope_phase":
                        segment_Y = GetClassY(os.path.join(dirpath, "exercises"), "Wide squat")
                        if segment_Y is None:
                            print("Error reading exercise phase:")
                            continue
                    for start in range(0, signal_length - window_size + 1, step_size):
                        if np.random.rand() < 0.5 and (dir != "transitions_envelope_phase" and dataset_type == "Reg"):
                            end = start + window_size // 2
                            segment_X = df.loc[start:end-1,channels]
                            if dataset_type == "Reg":
                                segment_Y = df.loc[end-1,["Exercise_Phase"]]
                            segment_x_interp = []
                            for channel in segment_X.T.values:
                                def_length = len(channel)
                                x_old = np.linspace(0, 1, def_length)
                                x_new = np.linspace(0, 1, window_size)
                                f = interp1d(x_old, channel, kind="linear")
                                segment_x_interp.append(f(x_new))
                            segment_x_interp = np.array(segment_x_interp).T
                            segment_X = segment_x_interp
                            # plt.plot(segment_X)
                            # plt.show()
                            if np.isnan(segment_X).any():
                                # print(file_name, start, end)
                                pass
                            else:
                                data_X[fname].append(segment_X)
                                data_Y[fname].append(segment_Y)

                        end = start + window_size
                        segment_X = df.loc[start:end-1,channels].to_numpy()
                        if dataset_type == "Reg":
                            segment_Y = df.loc[end - 1, ["Exercise_Phase"]]
                        if np.isnan(segment_X).any():
                            print(file_name,start,end)
                        else:
                            data_X[fname].append(segment_X)
                            data_Y[fname].append(segment_Y)
                        # print(segment_X.shape)

os.makedirs(os.path.join(save_path,save_dir_X),exist_ok=True)
os.makedirs(os.path.join(save_path,save_dir_Y),exist_ok=True)

x = np.array(data_X[fname].copy())
x = x.reshape(-1,20)

print(len(data_X["Wide squat"]))
print(len(data_Y["Wide squat"]))

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