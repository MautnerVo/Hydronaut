"""
Kod vytvoří csv s obálkou a fází cvičení využitím Emg_Envelope.py a exercise_phase.py
"""

import os
import pandas as pd
from exercise_phase import get_exercise_phase
from emg_envelope import envelope_emg
import matplotlib.pyplot as plt

PATH = r"Y:\Datasets\Fyzio"
new_dir = "signals_envelope_phase"


for dirpath, dirnames, filenames in os.walk(PATH):
    for dir in dirnames:
        if(dir == "exercises_signals"):
            path = os.path.join(dirpath,new_dir)
            os.makedirs(path,exist_ok=True)
            for file in os.listdir(os.path.join(dirpath,dir)):
                file_name = os.path.splitext(file)[0]
                df = pd.read_csv(os.path.join(dirpath,dir,file))
                exercise_phase = get_exercise_phase(df,file_name)
                try:
                    if exercise_phase is  not None:
                        envelope = envelope_emg(df)
                        df['Biceps_EMG_Envelope'] = envelope[0]
                        df['Triceps_EMG_Envelope'] = envelope[1]
                        df['Rectus_EMG_Envelope'] = envelope[2]
                        df['Gastrocnemious_EMG_Envelope'] = envelope[3]
                        df['Exercise_Phase'] = exercise_phase
                        df.to_csv(os.path.join(path, file), index=False)
                        plt.title(file)
                        plt.plot(exercise_phase)
                        plt.show()
                except:
                    print(dir,file)


