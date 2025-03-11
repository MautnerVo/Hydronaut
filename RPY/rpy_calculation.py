import calculations
import plot_euler as peu
from skinematics import imus
import numpy as np
import pandas as pd
import time

rate = 100
file = r"IMU/MT_00130934_003-000.csv"
df = pd.read_csv(file,sep="\t")

Acc_X = df["Acc_X"]
Acc_Y = df["Acc_Y"]
Acc_Z = df["Acc_Z"]

Gyr_X = df["Gyr_X"]
Gyr_Y = df["Gyr_Y"]
Gyr_Z = df["Gyr_Z"]

Mag_X = df["Mag_X"]
Mag_Y = df["Mag_Y"]
Mag_Z = df["Mag_Z"]

mat = np.array([[df["Mat[0][0]"],df["Mat[0][1]"],df["Mat[0][2]"]],[df["Mat[1][0]"],df["Mat[1][1]"],df["Mat[1][2]"]],[df["Mat[2][0]"],df["Mat[2][1]"],df["Mat[2][2]"]]]).T

acc = np.array([Acc_X,Acc_Y,Acc_Z]).T
gyro = np.array([Gyr_X,Gyr_Y,Gyr_Z]).T
mag = np.array([Mag_X,Mag_Y,Mag_Z]).T

tau = [0.6,0.6,0.6]
D = [0.2, 0.2, 0.2]

Q_k = np.eye(7) * 1e-4
R_k = np.eye(7) * 1e-3

start = time.time()
kOut = imus.kalman(rate, acc, gyro, mag, D=D, tau=tau, Q_k=Q_k, R_k=R_k)
t = np.arange(len(acc)) / rate
qOut = calculations.estimate_orientation(acc, gyro, t)

end = time.time()
print(end - start)

kroll_list = []
kpitch_list = []
kyaw_list = []

roll_list = []
pitch_list = []
yaw_list = []

mroll_list = []
mpitch_list = []
myaw_list = []

for q, k in zip(qOut, kOut):
    k0, k1, k2, k3 = k
    q0, q1, q2, q3 = q.w, q.x, q.y, q.z

    kroll, kpitch, kyaw = calculations.rpy_quaternion(k0, k1, k2, k3)
    kroll_list.append(kroll)
    kpitch_list.append(kpitch)
    kyaw_list.append(kyaw)

    roll,pitch,yaw = calculations.rpy_quaternion(q0, q1, q2, q3)
    roll_list.append(roll)
    pitch_list.append(pitch)
    yaw_list.append(yaw)

for m in mat:
    mroll,mpitch,myaw = calculations.rpy_matrix(m[0][0], m[1][0], m[2][0], m[2][1], m[2][2])
    mroll_list.append(mroll)
    mpitch_list.append(mpitch)
    myaw_list.append(myaw)

peu.plot_euler(kroll_list[:5000],kpitch_list[:5000],kyaw_list[:5000])
peu.plot_euler(roll_list[:5000],pitch_list[:5000],yaw_list[:5000])
peu.plot_euler(mroll_list[:5000],mpitch_list[:5000],myaw_list[:5000])