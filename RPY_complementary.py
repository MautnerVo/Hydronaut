from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import filtfilt, butter
from quaternion import quaternion, from_rotation_vector, rotate_vectors
import time

def estimate_orientation(a, w, t, alpha=0.9, g_ref=(0., 0., 1.),
                         theta_min=1e-6, highpass=.01, lowpass=.05):
    """ Estimate orientation with a complementary filter.
    Fuse linear acceleration and angular velocity measurements to obtain an
    estimate of orientation using a complementary filter as described in
    `Wetzstein 2017: 3-DOF Orientation Tracking with IMUs`_
    .. _Wetzstein 2017: 3-DOF Orientation Tracking with IMUs:
    https://pdfs.semanticscholar.org/5568/e2100cab0b573599accd2c77debd05ccf3b1.pdf
    Parameters
    ----------
    a : array-like, shape (N, 3)
        Acceleration measurements (in arbitrary units).
    w : array-like, shape (N, 3)
        Angular velocity measurements (in rad/s).
    t : array-like, shape (N,)
        Timestamps of the measurements (in s).
    alpha : float, default 0.9
        Weight of the angular velocity measurements in the estimate.
    g_ref : tuple, len 3, default (0., 0., 1.)
        Unit vector denoting direction of gravity.
    theta_min : float, default 1e-6
        Minimal angular velocity after filtering. Values smaller than this
        will be considered noise and are not used for the estimate.
    highpass : float, default .01
        Cutoff frequency of the high-pass filter for the angular velocity as
        fraction of Nyquist frequency.
    lowpass : float, default .05
        Cutoff frequency of the low-pass filter for the linear acceleration as
        fraction of Nyquist frequency.
    Returns
    -------
    q : array of quaternions, shape (N,)
        The estimated orientation for each measurement.
    """

    # initialize some things
    N = len(t)
    dt = np.diff(t)
    g_ref = np.array(g_ref)
    q = np.ones(N, dtype=quaternion)

    # get high-passed angular velocity
    w = filtfilt(*butter(5, highpass, btype='high'), w, axis=0)
    w[np.linalg.norm(w, axis=1) < theta_min] = 0
    q_delta = from_rotation_vector(w[1:] * dt[:, None])

    # get low-passed linear acceleration
    a = filtfilt(*butter(5, lowpass, btype='low'), a, axis=0)

    for i in range(1, N):

        # get rotation estimate from gyroscope
        q_w = q[i - 1] * q_delta[i - 1]

        # get rotation estimate from accelerometer
        v_world = rotate_vectors(q_w, a[i])
        n = np.cross(v_world, g_ref)
        phi = np.arccos(np.dot(v_world / np.linalg.norm(v_world), g_ref))
        q_a = from_rotation_vector(
            (1 - alpha) * phi * n[None, :] / np.linalg.norm(n))[0]

        # fuse both estimates
        q[i] = q_a * q_w

    return q


file = "2024-09-25T14-00.mat"

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
t = np.arange(len(acc)) / rate
start = time.time()
qOut = estimate_orientation(acc, gyro, t)
end = time.time()

print(end-start)
roll_list = []
pitch_list = []
yaw_list = []

roll = 0
pitch = 0
yaw = 0


for q in qOut:
    q0, q1, q2, q3 = np.array([q.x, q.y, q.z, q.w])

    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = 2 * math.atan2(math.sqrt(1+ 2 *(q0*q2-q1*q3)),math.sqrt(1-2*(q0*q2-q1*q3))) - math.pi/2
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    roll_list.append(math.degrees(roll))
    pitch_list.append(math.degrees(pitch))
    yaw_list.append(math.degrees(yaw))

data = np.array([roll_list, pitch_list, yaw_list]).T

df = pd.DataFrame(data)
df.columns = ["Roll", "Pitch", "Yaw"]
df.to_csv("orientace_complementary.csv",index=False)

plt.subplot(3, 1, 1)
plt.title("Roll")
plt.xlabel("cas")
plt.ylabel("hodnota stupne")
plt.plot(roll_list)
plt.subplot(3, 1, 2)
plt.title("Pitch")
plt.xlabel("cas")
plt.ylabel("hodnota stupne")
plt.plot(pitch_list)
plt.subplot(3, 1, 3)
plt.title("Yaw")
plt.xlabel("cas")
plt.ylabel("hodnota stupne")
plt.suptitle("Komplementární filtr")
plt.plot(yaw_list)
plt.show()
