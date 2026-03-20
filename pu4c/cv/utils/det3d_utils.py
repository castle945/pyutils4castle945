# 三维目标检测相关工具函数
import numpy as np

def boxes3d_to_corners_and_lines(boxes3d: np.ndarray, show_heading: bool = True) -> np.ndarray:
    """批量计算边界框数组的角点与线框，要求输入的边界框物体坐标系为前左上
    Args:
        boxes3d (ndarray(N, 7))
        show_heading (bool): 是否添加头部交叉线
    Returns:
        corners (ndarray(N*8, 3)): 所有边界框的角点，共 N*8 个
        lines (ndarray(N*12 or N*14, 2)): 所有边界框的线条，共有 N*12 或 N*14 条（每个线条就是两个角点索引），给定框 i 的线条 j，lines[i*12+j] = [x,y] 表示从角点 corners[x] 到角点 corners[y] 的连线
    Notes:
        - 此函数输出的角点顺序如下图，假设物体坐标系为前左上，即 X 轴正向处偏航角为 0
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
        - numpy >=1.20 对于 (N,3)@(3,3) 不再自动广播，必须构建 (N,3,3)
    """
    N, (x, y, z, l, w, h, yaw) = boxes3d.shape[0], boxes3d.T

    # 8 个角点在原点处的局部坐标
    corners = (np.array([
        [-l, -w, -h], [ l, -w, -h], [-l,  w, -h], [-l, -w,  h],
        [ l,  w,  h], [-l,  w,  h], [ l, -w,  h], [ l,  w, -h],
    ]) * 0.5).transpose(2, 0, 1) # (8, 3, N) -> (N, 8, 3)
    # 绕 Z 轴旋转偏航角的旋转矩阵
    sina, cosa = np.sin(yaw), np.cos(yaw)
    zeros, ones = np.zeros_like(cosa), np.ones_like(cosa)
    Rz_T = np.stack([
        np.stack([cosa,  sina, zeros], axis=-1),
        np.stack([-sina, cosa, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1),
    ], axis=1)
    corners = corners @ Rz_T # (N, 8, 3) @ (N, 3, 3)，会将 N 作为批量维度，后两维视为矩阵做乘法得到 (N, 8, 3)
    # 平移为全局坐标
    corners += boxes3d[:, None, 0:3]

    # 单个边界框的局部线条，值为框内相对索引 (12 or 14, 2)
    lines = [
        [0, 2], [0, 3], [2, 5], [3, 5],
        [0, 1], [3, 6], [5, 4], [2, 7],
        [1, 6], [1, 7], [7, 4], [4, 6],
    ]
    if show_heading:
        lines.extend([[1, 4], [6, 7]])
    # 生成 N 个框的索引偏移量，加到相对索引上得到全局索引，整体逻辑相当于 np.array([np.array(lines) + i*8 for i in range(N)])
    lines = np.array(lines)[None, :] + np.arange(N)[:, None, None] * 8    # (N, 12 or 14, 2)

    return corners.reshape(-1, 3), lines.reshape(-1, 2)

# copyed from OpenPCDet
def limit_period(val, offset: float = 0.5, period: float = 2*np.pi):
    """将周期值映射到 [-offset*period, (1-offset) * period]
    Args:
        val (ndarray(N,))
    Notes:
        - 三维目标检测中，框坐标系转换后偏航角取值范围可能大于 2pi，可以将其调整为[-pi,pi]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val
def mask_points_and_boxes_outside_range(points: np.ndarray, limit_range: np.ndarray, boxes3d: np.ndarray = None):
    point_mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    box_mask = ((boxes3d[:, :3] >= limit_range[:3]) & (boxes3d[:, :3]  <= limit_range[3:6])).all(axis=-1) if boxes3d is not None else None
    return point_mask, box_mask
