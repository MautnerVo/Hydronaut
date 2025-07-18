"""
Vytvoření emg obálky
Přidání obálky do csv souboru
"""
# import neurokit2 as nk
# print(nk.__version__)
# import scipy
# print(scipy.__version__)

from scipy.signal import butter, filtfilt
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

# emg = emg.astype(float)
def custom_highpass(signal, cutoff=10, fs=200, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered


def EnvelopeEMG(dataframe, full=False):
    """

    :param full: true | false vrací všechny signály | vrací pouze obálku emg signálu
    :param dataframe: Pandas dataframe načtený z csv cvičení cvičení

    :return: vrací obálku emg signálu

    """
    emg_triceps = dataframe['Triceps_EMG']
    emg_biceps = dataframe['Biceps_EMG']
    emg_rectus = dataframe['Rectus_EMG']
    emg_gastrocnemious = dataframe['Gastrocnemious_EMG']


    emg_triceps_clean = custom_highpass(emg_triceps)
    emg_triceps_signals, info = nk.emg_process(emg_triceps_clean, sampling_rate=200,lowcut=20, highcut=80,
                                       method_cleaning="none")

    emg_biceps_clean = custom_highpass(emg_biceps)
    emg_biceps_signals, info = nk.emg_process(emg_biceps_clean, sampling_rate=200, lowcut=20, highcut=80,
                                       method_cleaning="none")

    emg_rectus_clean = custom_highpass(emg_rectus)
    emg_rectus_signals, info = nk.emg_process(emg_rectus_clean, sampling_rate=200, lowcut=20, highcut=80,
                                       method_cleaning="none")

    emg_gastrocnemious_clean = custom_highpass(emg_gastrocnemious)
    emg_gastrocnemious_signals, info = nk.emg_process(emg_gastrocnemious_clean, sampling_rate=200, lowcut=20, highcut=80,
                                       method_cleaning="none")
    # nk.emg_plot(emg_signals ,info)
    #
    # plt.show()
    if full:
        return (emg_biceps_signals,emg_triceps_signals, emg_rectus_signals, emg_gastrocnemious_signals)

    return (emg_biceps_signals.iloc[:,2],emg_triceps_signals.iloc[:,2],emg_rectus_signals.iloc[:,2],emg_gastrocnemious_signals.iloc[:,2])

if __name__ == '__main__':
    df = pd.read_csv(r"Y:\Datasets\Fyzio\2025-03-07\1\exercises_signals\Standard squat.csv")
    signals = EnvelopeEMG(df)
    df['Biceps_EMG_Envelope'] = signals[0]
    df['Triceps_EMG_Envelope'] = signals[1]
    df['Rectus_EMG_Envelope'] = signals[2]
    df['Gastrocnemious_EMG_Envelope'] = signals[3]
    df.to_csv("Test_Squat.csv", index=False)