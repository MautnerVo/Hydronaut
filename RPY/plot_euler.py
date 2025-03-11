import matplotlib.pyplot as plt

def plot_euler(roll,pitch,yaw):
    plt.subplot(311)
    plt.plot(roll)
    plt.subplot(312)
    plt.plot(pitch)
    plt.subplot(313)
    plt.plot(yaw)
    plt.show()