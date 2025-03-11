import pandas as pd
from scipy.stats import pearsonr
import numpy as np
kalman = pd.read_csv("../orientace_Kalman.csv")
komplementarni = pd.read_csv("../orientace_Komplementarni.csv")


def normalize(signal):
    return signal - np.mean(signal)

kR = normalize(kalman["Roll"])
kP = normalize(kalman["Pitch"])
kY = normalize(kalman["Yaw"])

cP = normalize(komplementarni["Pitch"])
cY = normalize(komplementarni["Yaw"])
cR = normalize(komplementarni["Roll"])

Roll_corr,Rp = pearsonr(kR,cR)
Pitch_corr,Pp = pearsonr(kP,cP)
Yaw_corr,Yp = pearsonr(kY,cY)

print(Roll_corr)
print(Pitch_corr)
print(Yaw_corr)