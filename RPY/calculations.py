import math
from scipy.signal import filtfilt, butter
from quaternion import quaternion, from_rotation_vector, rotate_vectors
import numpy as np

def roll_matrix(r21, r22):
    return math.atan2(r21,r22)

def pitch_matrix(r20):
    return math.asin(-r20)

def yaw_matrix(r10,r00):
    return math.atan2(r10,r00)

def rpy_matrix(r00,r10,r20,r21,r22):
    return roll_matrix(r21,r22), pitch_matrix(r20),yaw_matrix(r10,r00)

def roll_quaternion(q0,q1,q2,q3):
    return math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

def pitch_quaternion(q0,q1,q2,q3):
    return math.asin(2 * (q0 * q2 - q3 * q1))

def yaw_quaternion(q0,q1,q2,q3):
    return math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

def rpy_quaternion(q0,q1,q2,q3):
    return roll_quaternion(q0,q1,q2,q3),pitch_quaternion(q0,q1,q2,q3),yaw_quaternion(q0,q1,q2,q3)

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