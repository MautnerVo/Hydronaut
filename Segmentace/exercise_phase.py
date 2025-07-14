"""
Vytvoření fází cvičení
"""

import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def get_exercise_phase(dataframe):
    """
    :param dataframe: Pandas dataframe načtený z csv cvičení cvičení
    :return: vrací signál fáze cvičení
    """
    quat = dataframe['Biceps_Q3']

    peaks, _ = find_peaks(-quat,height=0.8,distance=300)
    peak_distance = []

    for i in range(peaks.shape[0]-1):
        peak_distance.append(peaks[i+1] - peaks[i] - 1)

    step = []
    for i in peak_distance:
        step.append(100/i)

    exercise_phase = []

    for i in range(len(quat)):
        if(i < peaks[0]):
            exercise_phase.append(0)

    for index, rep in enumerate(step):
        for phase in range(peak_distance[index]):
            exercise_phase.append(rep * phase)

    while len(exercise_phase) < len(quat):
        exercise_phase.append(0)

    return exercise_phase

if __name__ == '__main__':
    df = pd.read_csv(r"Test_Squat.csv")
    exercise_phase = get_exercise_phase(df)

    df['ExercisePhase'] = exercise_phase
    df.to_csv("Test_Squat_full.csv", index=False)