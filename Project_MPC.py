import numpy as np


def rect_edge_point(width, height, phi):
    phi = np.asarray(phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    dx = np.where(cos_phi != 0, np.abs(width / 2 / cos_phi), np.inf)
    dy = np.where(sin_phi != 0, np.abs(height / 2 / sin_phi), np.inf)
    t = np.minimum(dx, dy)
    x = t * cos_phi
    y = t * sin_phi
    normal_x = np.where(dx < dy, np.copysign(1, -cos_phi), 0)
    normal_y = np.where(dx >= dy, np.copysign(1, -sin_phi), 0)
    tangential_x = np.where(dx < dy, 0, np.copysign(1, sin_phi))
    tangential_y = np.where(dx >= dy, 0, np.copysign(1, -cos_phi))
    normal = np.stack((normal_x, normal_y), axis=-1)
    tangential = np.stack((tangential_x, tangential_y), axis=-1)
    return x, y, normal, tangential

def get_B(x, y, normal, tangential):
    x = np.asarray(x)
    y = np.asarray(y)
    C = x.shape[0]
    J = np.zeros((C, 2, 3))
    J[:, 0, 0] = 1
    J[:, 1, 1] = 1
    J[:, 0, 2] = -y
    J[:, 1, 2] = x
    N = np.einsum('cij,cj->ci', J.transpose(0, 2, 1), normal)
    T = np.einsum('cij,cj->ci', J.transpose(0, 2, 1), tangential)
    B = np.hstack((N.T, T.T))
    return B

def rotmat(t):
    ct = np.cos(t)
    st = np.sin(t)
    return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

def angle_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def get_B_k(x, b, B_L):
    r = rotmat(x[2])
    rl = r @ B_L
    rlb = rl @ b
    rlb_augment = np.block([[rlb], [np.zeros((2, 4))]])
    return rlb_augment