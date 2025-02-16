from scipy.io import loadmat
from skinematics import imus
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

import time

file  = "2024-09-25T14-00.mat"


mat = loadmat(file)
EMG = mat["EMG"][0][0]

Acc_X = EMG["Sensor_2"]["Acceleration"][0][0]["X"][0][0]
Acc_Y = EMG["Sensor_2"]["Acceleration"][0][0]["Y"][0][0]
Acc_Z = EMG["Sensor_2"]["Acceleration"][0][0]["Z"][0][0]

Gyro_X = EMG["Sensor_2"]["Gyroscope"][0][0]["X"][0][0]
Gyro_Y = EMG["Sensor_2"]["Gyroscope"][0][0]["Y"][0][0]
Gyro_Z = EMG["Sensor_2"]["Gyroscope"][0][0]["Z"][0][0]

Magne_X = EMG["Sensor_2"]["Magnetometer"][0][0]["X"][0][0]
MAgne_Y = EMG["Sensor_2"]["Magnetometer"][0][0]["Y"][0][0]
Magne_Z = EMG["Sensor_2"]["Magnetometer"][0][0]["Z"][0][0]

acc = np.array([(Ax[1],Ay[1],Az[1])for Ax,Ay,Az in zip(Acc_X,Acc_Y,Acc_Z)])
gyro = np.array([(Gx[1],Gy[1],Gz[1])for Gx,Gy,Gz in zip(Gyro_X,Gyro_Y,Gyro_Z)])
mag = np.array([(Mx[1],My[1],Mz[1])for Mx,My,Mz in zip(Magne_X,MAgne_Y,Magne_Z)])

rate = 100

D = [0.4, 0.4, 0.4]
tau = [0.5, 0.5, 0.5]

start = time.time()
qOut = imus.kalman(rate, acc, gyro, mag, D=D, tau=tau)
end = time.time()

roll_list = []
pitch_list = []
yaw_list = []

for q in qOut:
    q0, q1, q2, q3 = q

    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = 2 * math.atan2(math.sqrt(1+ 2 *(q0*q2-q1*q3)),math.sqrt(1-2*(q0*q2-q1*q3))) - math.pi/2
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    roll_list.append(math.degrees(roll))
    pitch_list.append(math.degrees(pitch))
    yaw_list.append(math.degrees(yaw))

data = np.array([roll_list, pitch_list, yaw_list]).T

df = pd.DataFrame(data)
df.columns = ["Roll", "Pitch", "Yaw"]
df.to_csv("orientace_kalman.csv",index=False)

print(end-start)

plt.subplot(3,1,1)
plt.title("Roll")
plt.xlabel("cas")
plt.ylabel("hodnota stupne")
plt.plot(roll_list)
plt.subplot(3,1,2)
plt.title("Pitch")
plt.xlabel("cas")
plt.ylabel("hodnota stupne")
plt.plot(pitch_list)
plt.subplot(3,1,3)
plt.title("Yaw")
plt.xlabel("cas")
plt.ylabel("hodnota stupne")
plt.suptitle("Kalmanův Filtr")
plt.plot(yaw_list)
plt.show()


