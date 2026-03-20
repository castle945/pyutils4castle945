from typing import Any, Union, Optional, Dict, List, Tuple
import numpy as np

# 投影
def project_points_to_pixels(points: np.ndarray, image_shape: tuple, transform_mat: np.ndarray):
    """
    Args:
        transform_mat (ndarray(4, 4)): 激光雷达坐标系到像素坐标系的变换矩阵，它等于 相机内参矩阵 @ 激光雷达到相机坐标系的变换矩阵
    Notes:
        - 坐标变换公式:
            (1) 变换矩阵形式: y(4,N) = R(4,4)*x(4,N) 变换矩阵同理定义为右乘列向量的点坐标的增广，不一定是正交阵，往往对公式转置一下以适应点云输入，即 y.T(N,4) = x.T(N,4)*R.T(4,4)
            (2) 旋转矩阵加平移向量形式: y(3,N) = R(3,3)*x(3,N) + t(3,1) 平移向量为列向量，旋转矩阵也定义为右乘列向量的点坐标，旋转矩阵为正交阵具有转置等于逆和行列式为 1 的特性
        - 老版本 numpy 数组转置后内存不连续，从而导致计算错误，故需要再转成内存连续后再计算
    """
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32))) # [N, 4]
    points_pixel = points_hom @ np.ascontiguousarray(transform_mat.T)[:, :3]
    
    pixels_depth = points_pixel[:, 2]
    pixels = (points_pixel[:, :2].T / points_pixel[:, 2]).T # (N, 2)[col, row]

    # remove points outside the image
    mask = pixels_depth > 0
    mask = np.logical_and(mask, pixels[:, 0] > 0)
    mask = np.logical_and(mask, pixels[:, 0] < image_shape[1])
    mask = np.logical_and(mask, pixels[:, 1] > 0)
    mask = np.logical_and(mask, pixels[:, 1] < image_shape[0])

    return pixels, pixels_depth, mask
def project_pixels_to_points(pixels: np.ndarray, depth: np.ndarray, transform_mat: np.ndarray):
    """
    Args:
        pixels (ndarray(N, 2)[x,y]): 像素坐标
        depth (ndarray(N,)): 像素深度
    """
    N = depth.shape[0]
    points_cam = np.zeros((N, 3))
    points_cam[:, 2] = depth
    points_cam[:, :2] = pixels * depth[:, np.newaxis].repeat(2, axis=1)

    points_cam_hom = np.hstack((points_cam, np.ones((N, 1), dtype=np.float32))) # [N, 4]
    points_hom = points_cam_hom @ np.ascontiguousarray(transform_mat.T)
    return points_hom[:, :3]
def project_points_to_pixels_cv2(points: np.ndarray, image_shape: tuple, lidar2cam_mat: np.ndarray, intrinsics_4x4, dist_coeffs):
    """如果带畸变参数，可以调 cv2.projectPoints 进行处理，输入相机坐标系下的点，输出二维像素坐标
    Args:
        lidar2cam_mat: 4x4 激光雷达到相机坐标系的变换矩阵
        intrinsics_4x4: 相机内参矩阵
        dist_coeffs: 相机畸变参数
    """
    import cv2
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32))) # [N, 4]
    points_cam = points_hom @ lidar2cam_mat.T

    pixels_depth = points_cam[:, 2] # 像素深度即相机坐标系下的 z 
    rotation, translation = np.eye(3), np.zeros((3, 1))
    pixels, jac = cv2.projectPoints(points_cam[:, :3].T, rotation, translation, intrinsics_4x4[:3, :3], dist_coeffs)
    pixels = pixels.squeeze(axis=1)

    # remove points outside the image
    mask = pixels_depth > 0
    mask = np.logical_and(mask, pixels[:, 0] > 0)
    mask = np.logical_and(mask, pixels[:, 0] < image_shape[1])
    mask = np.logical_and(mask, pixels[:, 1] > 0)
    mask = np.logical_and(mask, pixels[:, 1] < image_shape[0])

    return pixels, pixels_depth, mask

def range_projection(points: np.ndarray, height: int, width: int, fov: tuple):
    """lidar_to_rangeview 的仅线性拉伸简化版
    Args:
        fov: [fov_up, fov_down]
    """
    depth = np.linalg.norm(points[:, :3], ord=2, axis=1) # 按行求二范数，即距离
    yaw, pitch = -np.arctan2(points[:, 1], points[:, 0]), np.arcsin(points[:, 2] / depth)

    fov, fov_down = np.sum(np.abs(fov)), abs(fov[1])
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # yaw=[-pi, pi] to [0.0, 1.0]
    proj_y = 1.0 - (pitch + fov_down) / fov     # pitch=[fov_up, fov_down] to [0.0, 1.0]
    proj_x *= width     # to [0.0, W]
    proj_y *= height    # to [0.0, H]

    # 坐标取整作为像素坐标
    proj_x = np.minimum(width - 1, np.floor(proj_x))
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # to [0, W-1]
    proj_y = np.minimum(height - 1, np.floor(proj_y))
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # to [0, H-1]

    range_image = np.full((height, width), -1, dtype=np.float32) # [H,W] range (-1 is no data)
    point_idx = np.full((height, width), -1, dtype=np.int32)     # [H,W] index (-1 is no data)
    range_image[proj_y, proj_x] = depth
    point_idx[proj_y, proj_x] = np.arange(depth.shape[0])

    return range_image, point_idx
def lidar_to_rangeview(
    points: np.ndarray,
    height: int,
    width: int,
    fov: List[float] = None,
    resolution: List[float] = None,
    fov_offset_down: float = None,
    max_depth: float = None,
    return_intensity: bool = False,
):
    """一般算法都是以线性拉伸的方式转 RV 图像，其实这就是一种柱坐标系体素栅格
    Args:
        fov (list(fov_up, fov_down, fov_left, fov_right)): 增加水平视场角范围使可用于非机械雷达
        resolution (list(res_y, res_x)): None 则以线性拉伸的方式，否则传入角度的分辨率，按实际填入（类似给定 voxel_size 的体素化）
        fov_offset_down: 对底部 fov 的偏移量，当可以刻意把显示范围扩大时，fov_offset_down 可以小于 fov_down 以使得画面居中
        return_intensity: 是否返回反射强度图像
    Returns:
        range_image: 浮点数深度值，-1 表示无效点
        point_idx: 像素坐标到原始点云点坐标的索引，point = points[point_idx[y, x]] 或 point_idx.reshape(-1) rv_points = points[point_idx[point_idx != -1]]
        intensity_image: 反射强度，-1 表示无效点
    Examples:
        (1) 线性拉伸方式 lidar_to_rangeview(points, height, width, fov=fov)  
        (2) 非线性拉伸方式 lidar_to_rangeview(points, height, width, resolution=resolution, fov_down=fov_down)  
    """
    depth = np.linalg.norm(points[:, :3], ord=2, axis=1) # 按行求二范数，即距离
    # 前左上坐标系，yaw 加负号反向一下以遵循从左到右递增的习惯
    yaw, pitch = -np.arctan2(points[:, 1], points[:, 0]), np.arcsin(points[:, 2] / depth)

    point_idx_raw = np.arange(points.shape[0])
    if resolution is None:
        assert (fov is not None)
        fov = fov if len(fov) == 4 else [fov[0], fov[1], -np.pi, np.pi]
        fov_height, fov_width = np.sum(np.abs(fov[:2])), np.sum(np.abs(fov[2:]))
        fov_up, fov_down, fov_left, fov_right = np.abs(fov)
        
        proj_x = yaw * width / fov_width + width / 2.0    # fov_width 线性拉伸到 width，并将 x 轴原点移动到图像中心列以符合前左上坐标系
        proj_y = -((pitch + fov_down) * height / fov_height) + height # + abs(fov_down) 将 patch 范围移动到 [0, +]

        proj_x = np.minimum(width - 1, np.floor(proj_x))
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # to [0, W-1]
        proj_y = np.minimum(height - 1, np.floor(proj_y))
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # to [0, H-1]
    else:
        assert (fov_offset_down is not None)
        (res_y, res_x), fov_offset_down = resolution, abs(fov_offset_down)

        proj_x = yaw / res_x + width / 2.0
        proj_y = -((pitch + fov_offset_down) / res_y) + height

        mask_in_x = np.logical_and(np.floor(proj_x) >=0, np.floor(proj_x) < width)
        mask_in_y = np.logical_and(np.floor(proj_y) >=0, np.floor(proj_y) < height)
        mask_in = np.logical_and(mask_in_x, mask_in_y)
        proj_x, proj_y = proj_x[mask_in].astype(np.int32), proj_y[mask_in].astype(np.int32)
        depth, points, point_idx_raw = depth[mask_in], points[mask_in], point_idx_raw[mask_in]

    range_image = np.full((height, width), -1, dtype=np.float32) # [H,W] range (-1 is no data)
    point_idx = np.full((height, width), -1, dtype=np.int32)     # [H,W] index (-1 is no data)
    # 按深度降序排序，则之后对投影到同一个像素点的多个激光点会自然取距离最近的点，类似最近回波模式
    indices = np.argsort(depth)[::-1]
    if max_depth is not None:
        depth[depth > max_depth] = -1 # 超出最远距离的点置为 -1 过滤掉
    proj_x, proj_y = proj_x[indices], proj_y[indices]
    range_image[proj_y, proj_x] = depth[indices]
    point_idx[proj_y, proj_x] = point_idx_raw[indices]

    if return_intensity:
        intensity_image = np.full((height, width), -1, dtype=np.float32)
        intensity_image[proj_y, proj_x] = points[indices][:, 3]
        return range_image, point_idx, intensity_image

    return range_image, point_idx
def rangeview_to_lidar(
    range_image: np.ndarray,
    intensity_image: np.ndarray = None,
    fov: List[float] = None,
    resolution: List[float] = None,
    fov_offset_down: float = None,
):
    height, width = range_image.shape
    proj_x, proj_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")

    if resolution is None:
        assert (fov is not None)
        fov = fov if len(fov) == 4 else [fov[0], fov[1], -np.pi, np.pi]
        fov_height, fov_width = np.sum(np.abs(fov[:2])), np.sum(np.abs(fov[2:]))
        fov_up, fov_down, fov_left, fov_right = np.abs(fov)

        yaw = -((proj_x - width/2.0)*fov_width / width)
        pitch = (fov_up - proj_y*fov_height / height)
    else:
        assert (fov_offset_down is not None)
        (res_y, res_x), fov_offset_down = resolution, abs(fov_offset_down)

        yaw = -((proj_x - width/2.0) * res_x)
        pitch = ((height - proj_y) * res_y) - fov_offset_down

    dirs = np.stack([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)], axis=-1)
    points = dirs * range_image.reshape(height, width, 1)
    if intensity_image is not None:
        points = np.concatenate([points, intensity_image.reshape(height, width, 1)], axis=2)

    num_features = 3 if intensity_image is None else 4
    points = points[np.where(range_image != -1)].reshape(-1, num_features)

    return points


# 采样
def farthest_point_sample(point, npoint):
    """
    Args:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)  # N个点中随机取1个点作为初始采样点
    for i in range(npoint):
        centroids[i] = farthest         # 新增一个采样点，迭代 npoint 次
        centroid = xyz[farthest, :]     # 以采样点为中心点，计算到其他点的距离之和
        dist = np.sum((xyz - centroid) ** 2, -1) # axis=-1按最高维度雷达，二维则按列加，即行和即x**2+y**2+z**2
        mask = dist < distance          # 偏离太远的点置 false 丢弃
        distance[mask] = dist[mask]     # 只为为 true 的点更新有效距离， python 允许以bool数组做下标掩膜
        farthest = np.argmax(distance, -1)  # 取离当前点有效距离最远的点作为下一个采样点
    point = point[centroids.astype(np.int32)]
    return point
