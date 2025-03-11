import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import os
file = "2024-09-25T14-00.mat"
mat = loadmat(file)

EMG = mat["EMG"][0][0]

Acc_X = EMG["Sensor_3"]["Acceleration"][0][0]["X"][0][0]
Acc_Y = EMG["Sensor_3"]["Acceleration"][0][0]["Y"][0][0]
Acc_Z = EMG["Sensor_3"]["Acceleration"][0][0]["Z"][0][0]

Gyro_X = EMG["Sensor_3"]["Gyroscope"][0][0]["X"][0][0]
Gyro_Y = EMG["Sensor_3"]["Gyroscope"][0][0]["Y"][0][0]
Gyro_Z = EMG["Sensor_3"]["Gyroscope"][0][0]["Z"][0][0]

Magne_X = EMG["Sensor_3"]["Magnetometer"][0][0]["X"][0][0]
Magne_Y = EMG["Sensor_3"]["Magnetometer"][0][0]["Y"][0][0]
Magne_Z = EMG["Sensor_3"]["Magnetometer"][0][0]["Z"][0][0]

acc = np.array([(Ax[1],Ay[1],Az[1])for Ax,Ay,Az in zip(Acc_X,Acc_Y,Acc_Z)])
gyro = np.array([(Gx[1],Gy[1],Gz[1])for Gx,Gy,Gz in zip(Gyro_X,Gyro_Y,Gyro_Z)])
mag = np.array([(Mx[1],My[1],Mz[1])for Mx,My,Mz in zip(Magne_X,Magne_Y,Magne_Z)])

def AccPlot():
    plt.subplot(311)
    plt.plot([x[0] for x in acc])
    plt.subplot(312)
    plt.plot([y[1] for y in acc])
    plt.subplot(313)
    plt.plot([z[2] for z in acc])
    plt.show()

def GyroPlot():
    plt.subplot(311)
    plt.plot([x[0] for x in gyro])
    plt.subplot(312)
    plt.plot([y[1] for y in gyro])
    plt.subplot(313)
    plt.plot([z[2] for z in gyro])
    plt.show()

def MagPlot():
    plt.subplot(311)
    plt.plot([x[0]for x in mag])
    plt.subplot(312)
    plt.plot([y[1]for y in mag])
    plt.subplot(313)
    plt.plot([z[2]for z in mag])
    plt.show()

# print([z[2]for z in mag])
MagPlot()
AccPlot()
GyroPlot()

df_acc = pd.DataFrame(acc)
df_gyro = pd.DataFrame(gyro)
df_mag = pd.DataFrame(mag)

os.makedirs('IMU_results', exist_ok=True)
df_acc.to_csv(r"IMU_results/Acc.csv",header=False,index=False)
df_gyro.to_csv(r"IMU_results/Gyro.csv",header=False,index=False)
df_mag.to_csv(r"IMU_results/Mag.csv",header=False,index=False)