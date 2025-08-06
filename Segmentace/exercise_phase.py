"""
Vytvoření fází cvičení
"""

import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def normalize(signal):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return signal

def get_exercise_phase(dataframe,exercise,plot=False):
    """
    :param plot: True | False ukáže graf fáze
    :param exercise: jmeno vykonavaneho cviku
    :param dataframe: Pandas dataframe načtený z csv cvičení cvičení
    :return: vrací signál fáze cvičení
    """
    signal = np.array([])
    peak = np.array([])
    exercise_phase = []
    peak_distance = []


    if exercise == "Standart Squad":
        signal = normalize(np.array(dataframe['Biceps_Q3']))
        peaks, _ = find_peaks(-signal,height=0.8,distance=300)
    elif exercise == "Lunge (left leg forward)":
        signal = normalize(np.array(dataframe['Rectus_Q2']))
        peak, _ = find_peaks(-signal,height=0.4,distance=200)
    elif exercise == "Lunge (right leg forward)":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q3']))
        peak, _ = find_peaks(-signal, height=0.65, distance=400)
    elif exercise == "Arm rotations":
        pass
    elif exercise == "Burpees":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q1']))
        peak, _ = find_peaks(signal,height=-0.2, distance=500)
    elif exercise == "High knee run":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q3']))
        peak, _ = find_peaks(signal,height=-0.1, distance=100)
    elif exercise == "Jump squats":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q1']))
        peak, _ = find_peaks(-signal,height=0.5, distance=100)
    elif exercise == "Push-ups":
        signal = normalize(np.array(dataframe['Biceps_Q4']))
        peak, _ = find_peaks(signal,prominence=0.2,distance=300)
    elif exercise == "Triceps push-ups":
        signal = normalize(np.array(dataframe['Triceps_Q4']))
        peak, _ = find_peaks(-signal,prominence=0.4,distance=300)
    elif exercise == "Wide squat":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q4']))
        peak , _ = find_peaks(signal,prominence=0.3,distance=300)
    else:
        return None

    if peak.shape[0] == 0:
        return None

    for i in range(peak.shape[0]-1):
        peak_distance.append(peak[i+1] - peak[i] - 1)

    step = []
    for i in peak_distance:
        step.append(100/i)



    for i in range(len(signal)):
        if i < peak[0]:
            exercise_phase.append(0)

    for index, rep in enumerate(step):
        for phase in range(peak_distance[index]):
            exercise_phase.append(rep * phase)

    while len(exercise_phase) < len(signal):
        exercise_phase.append(0)

    if plot:
        plt.subplot(2,1,1)
        plt.plot(signal)
        plt.plot(peak, signal[peak], 'x')
        plt.subplot(2,1,2)
        plt.plot(exercise_phase)
        plt.show()

    return exercise_phase

if __name__ == '__main__':
    df = pd.read_csv(r"Y:\Datasets\Fyzio\2025-03-07\3\exercises_signals\Wide squat.csv")
    exercisePhase = get_exercise_phase(df, "Wide squat",True)

    # df['ExercisePhase'] = exercise_phase
    # df.to_csv("Test_Squat_full.csv", index=False)