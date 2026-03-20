import numpy as np
import os

def read_points(filepath, num_features: int = 4) -> np.ndarray:
    filetype = os.path.splitext(filepath)[-1]
    if filetype == ".bin":
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, num_features)
    elif filetype == ".pcd" or filetype == '.ply':
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(filepath).points)
    elif filetype == ".npy":
        points = np.load(filepath)
    elif filetype == ".pkl" or filetype == ".gz": # '.pkl.gz'
        import pandas as pd
        points = pd.read_pickle(filepath).to_numpy()
    elif filetype == ".txt":
        points = np.loadtxt(filepath, dtype=np.float32).reshape(-1, num_features)
    else:
        raise TypeError("unsupport file type")

    return points

def transform_matrix(rotation_mat: np.ndarray, translation: np.ndarray, inverse: bool = False) -> np.ndarray:
    """传入变换矩阵中拆解的旋转矩阵和平移向量，返回变换矩阵或变换矩阵的逆，要求变换矩阵只由刚体变换计算(即不包括含相机内参矩阵的计算)得到而不能对任意 4x4 矩阵的拆解求逆"""
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation_mat.T
        trans = np.transpose(-translation)
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation_mat
        tm[:3, 3] = np.transpose(translation)

    return tm

def pad33to44(mat: np.ndarray) -> np.ndarray:
    tm = np.eye(4, dtype=mat.dtype)
    tm[:3, :3] = mat
    return tm