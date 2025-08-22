"""
Kod vytvoří csv s obálkou a fází cvičení využitím Emg_Envelope.py a exercise_phase.py
"""

import os
import numpy as np
import pandas as pd
from exercise_phase import get_exercise_phase
from emg_envelope import envelope_emg
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

PATH = r"Y:\Datasets\Fyzio"
new_dir = "signals_envelope_phase"
new_dir_envelopes = "signals_envelope"

sorted_collumns = ["Sample","Biceps_EMG","Biceps_EMG_Envelope","Biceps_Q1","Biceps_Q2","Biceps_Q3","Biceps_Q4",
                   "Triceps_EMG","Triceps_EMG_Envelope","Triceps_Q1","Triceps_Q2","Triceps_Q3","Triceps_Q4",
                   "Gastrocnemious_EMG","Gastrocnemious_EMG_Envelope","Gastrocnemious_Q1","Gastrocnemious_Q2","Gastrocnemious_Q3","Gastrocnemious_Q4",
                   "Rectus_EMG","Rectus_EMG_Envelope","Rectus_Q1","Rectus_Q2","Rectus_Q3","Rectus_Q4"]

def signal_FFT(signal):
    N = len(signal)
    signal_fft = fft(signal)
    freq = fftfreq(N, 1/200)
    idx = np.arange(N // 2)
    xf = freq[idx]
    yf = 2.0 / N * np.abs(signal_fft[idx])

    return xf,yf

def show_graph():
    x_range = range(start_end[0], start_end[1])
    plt.subplot(3, 1, 1)
    plt.title(os.path.join(path, file))
    plt.plot(x_range, exercise_phase[start_end[0]:start_end[1]])
    plt.ylabel("faze cviku")
    plt.subplot(3, 1, 2)
    plt.plot(x_range, signal[start_end[0]:start_end[1]])
    plt.plot(peak, signal[peak], 'x')
    plt.ylabel("oriznuty signal")
    plt.subplot(3, 1, 3)
    plt.plot(signal)
    plt.plot(peak, signal[peak], 'x')
    plt.ylabel("cely signal")
    plt.show()

for dirpath, dirnames, filenames in os.walk(PATH):
    for dir in dirnames:
        if(dir == "exercises_signals"):
            path = os.path.join(dirpath,new_dir)
            path_envelopes = os.path.join(dirpath,new_dir_envelopes)
            os.makedirs(path,exist_ok=True)
            os.makedirs(path_envelopes,exist_ok=True)
            for file in os.listdir(os.path.join(dirpath,dir)):
                file_name = os.path.splitext(file)[0]
                df = pd.read_csv(os.path.join(dirpath,dir,file))
                print(df.shape)
                try:
                    exercise_phase,start_end, signal, peak = get_exercise_phase(df,file_name)
                    if exercise_phase is  not None:
                        envelope = envelope_emg(df)
                        df['Biceps_EMG_Envelope'] = envelope[0]
                        df['Triceps_EMG_Envelope'] = envelope[1]
                        df['Rectus_EMG_Envelope'] = envelope[2]
                        df['Gastrocnemious_EMG_Envelope'] = envelope[3]
                        df = df.loc[start_end[0]:start_end[1]-1,:]
                        df = df[sorted_collumns]
                        df['Exercise_Phase'] = exercise_phase[start_end[0]:start_end[1]]
                        df.to_csv(os.path.join(path, file), index=False)
                        show_graph()
                except Exception as e:
                    print(e)
                    try:
                        if not os.path.exists(os.path.join(path_envelopes,file)):
                            envelope = envelope_emg(df)
                            df['Biceps_EMG_Envelope'] = envelope[0]
                            df['Triceps_EMG_Envelope'] = envelope[1]
                            df['Rectus_EMG_Envelope'] = envelope[2]
                            df['Gastrocnemious_EMG_Envelope'] = envelope[3]
                            df = df[sorted_collumns]
                            df.to_csv(os.path.join(path_envelopes, file), index=False)
                    except Exception as e:
                        print(e)
                        print(file)

