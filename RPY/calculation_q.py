import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

file = "walking normal_Processed.csv"

df = pd.read_csv(file)

Q = np.array([df['Q0_LeftFoot'],df['Q1_LeftFoot'],df['Q2_LeftFoot'],df['Q3_LeftFoot']]).T

roll_list = []
pitch_list = []
yaw_list = []

for q in Q:
    q0, q1, q2, q3 = q

    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    pitch = math.asin(2 * (q0 * q2 - q3 * q1))
    # pitch = 2 * math.atan2(math.sqrt(1+ 2 *(q0*q2-q1*q3)),math.sqrt(1-2*(q0*q2-q1*q3))) - math.pi/2 #dava stejne vysledky
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    roll_list.append(math.degrees(roll))
    pitch_list.append(math.degrees(pitch))
    yaw_list.append(math.degrees(yaw))

data = np.array([roll_list, pitch_list, yaw_list]).T

df = pd.DataFrame(data)
df.columns = ["Roll", "Pitch", "Yaw"]
df.to_csv("orientace_komplementarni.csv",index=False)

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