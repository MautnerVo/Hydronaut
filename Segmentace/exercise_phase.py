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
    start_end = (None, None)

    exceptions = ["Triceps push-ups","Standard squat"]

    limit = 1250 if exercise in exceptions else 850
    limit = 3000 if exercise == "Burpees" else limit
    limit = 500 if exercise == "Push ups" else limit

    if exercise == "Standard squat":
        signal = normalize(np.array(dataframe['Biceps_Q3']))
        peak, _ = find_peaks(-signal,prominence=0.7)
    elif exercise == "Lunge (left leg forward)":
        signal = normalize(np.array(dataframe['Rectus_Q2']))
        peak, _ = find_peaks(-signal,prominence=0.7)
    elif exercise == "Lunge (right leg forward)":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q3']))
        peak, _ = find_peaks(-signal, prominence=0.7)
    elif exercise == "Burpees":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q1']))
        peak, _ = find_peaks(-signal,prominence=0.8,distance=400)
    elif exercise == "High knee run":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q3']))
        peak, _ = find_peaks(signal,prominence=0.5)
    elif exercise == "Jump squats":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q1']))
        peak, _ = find_peaks(-signal,prominence=0.6)
    elif exercise == "Push-ups":
        signal = normalize(np.array(dataframe['Biceps_Q4']))
        peak, _ = find_peaks(signal,prominence=0.25)
    elif exercise == "Triceps push-ups":
        signal = normalize(np.array(dataframe['Triceps_Q4']))
        peak, _ = find_peaks(-signal,prominence=0.75)
    elif exercise == "Wide squat":
        signal = normalize(np.array(dataframe['Gastrocnemious_Q1']))
        peak , _ = find_peaks(signal,prominence=0.5)
    else:
        return None

    if peak.shape[0] == 0:
        return None

    for i in range(peak.shape[0]-1):
        if peak[i+1] - peak[i] < limit:
            peak_distance.append(peak[i+1] - peak[i] - 1)
        else:
            peak = peak[:i+1]
            break

    start_end = (peak[0], peak[-1])
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

    return exercise_phase,start_end, signal, peak,

if __name__ == '__main__':
    df = pd.read_csv(r"Y:\Datasets\Fyzio\2025-03-21\12\exercises_signals\Wide squat.csv")
    exercisePhase = get_exercise_phase(df, "Wide squat",True)

    # df['ExercisePhase'] = exercise_phase
    # df.to_csv("Test_Squat_full.csv", index=False)