import numpy as np
# rgb 排列组合 27 种去除黑白，并把七色环颜色前移
det25cls_colormap = [
    [0.0, 1.0, 0.0], # 绿
    [1.0, 0.0, 0.0], # 红
    [1.0, 0.5, 0.0], # 橙
    [1.0, 1.0, 0.0], # 黄
    [0.0, 0.0, 1.0], # 蓝
    [0.0, 1.0, 1.0], # 靛
    [1.0, 0.0, 1.0], # 紫
    [1.0, 1.0, 0.5],
    [1.0, 0.5, 1.0],
    [1.0, 0.5, 0.5],
    [1.0, 0.0, 0.5],
    [0.5, 1.0, 1.0],
    [0.5, 1.0, 0.5],
    [0.5, 1.0, 0.0],
    [0.5, 0.5, 1.0],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 1.0],
    [0.5, 0.0, 0.5],
    [0.5, 0.0, 0.0],
    [0.0, 1.0, 0.5],
    [0.0, 0.5, 1.0],
    [0.0, 0.5, 0.5],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
]

def plot_colormap(colormap, save_path: str, strip_height: int = 10, width: int = 100):
    from matplotlib import pyplot as plt
    height = len(colormap) * strip_height
    img_grey = np.zeros((height, width, 1), dtype=np.uint8)
    color_idx = 0
    for start in range(0, height, strip_height):
        img_grey[start:start+strip_height] = color_idx
        color_idx += 1
    img_rgb = np.array(colormap)[img_grey.reshape(-1)].reshape(height, width, 3)

    plt.imsave(save_path, img_rgb)

# modified from getRainbowColor() in https://github.com/ros-visualization/rviz/blob/noetic-devel/src/rviz/default_plugin/point_cloud_transformers.cpp
def rviz_intensity_colormap(intensity: np.ndarray, in_norm: bool = None, out_norm: bool = False):
    """rviz 风格的点云反射率颜色映射，将反射率映射为 rgb 数组
    Args:
        intensity (ndarray(N,)): 反射率
        in_norm: None 则推断，True/False 则表示输入数据已经/没有归一化
    """
    if in_norm is None:
        in_norm = np.max(intensity) <= 1
    colors = np.zeros((intensity.shape[0], 3), dtype=np.float64)
    
    norm_i = intensity / (1.0 if in_norm else 255.0)
    value = 1.0 - norm_i
    h = value * 5.0 + 1.0
    i = np.floor(h).astype(np.int64)
    f = h - i
    mask = (i & 1) == 0 # if i is even
    f[mask] = 1 - f[mask]
    n = 1 - f
    mask0, mask1, mask2, mask3, mask4 = (i <= 1), (i == 2), (i == 3), (i == 4), (i >= 5)
    colors[mask0] = np.vstack((n[mask0] * 255, np.zeros_like(n[mask0]), np.ones_like(n[mask0]) * 255)).T
    colors[mask1] = np.vstack((np.zeros_like(n[mask1]), n[mask1] * 255, np.ones_like(n[mask1]) * 255)).T
    colors[mask2] = np.vstack((np.zeros_like(n[mask2]), np.ones_like(n[mask2]) * 255, n[mask2] * 255)).T
    colors[mask3] = np.vstack((n[mask3] * 255, np.ones_like(n[mask3]) * 255, np.zeros_like(n[mask3]))).T
    colors[mask4] = np.vstack((np.ones_like(n[mask4]) * 255, n[mask4] * 255, np.zeros_like(n[mask4]))).T

    # 另一种类似 CloudCompare 风格的颜色映射
    # n = intensity * (255.0 if in_norm else 1.0)
    # mask0, mask1, mask2, mask3 = (n <= 30), np.logical_and(n > 30, n <= 90), np.logical_and(n > 90, n <= 150), np.logical_and(n > 150, n <= 255)
    # colors[mask0] = np.vstack((np.zeros_like(n[mask0]), n[mask0] * 255 / 30.0, np.ones_like(n[mask0]) * 255)).T
    # colors[mask1] = np.vstack((np.zeros_like(n[mask1]), np.ones_like(n[mask1]) * 255, ((90 - n[mask1]) * 255) / 60.0)).T
    # colors[mask2] = np.vstack(((n[mask2] - 90) * 255 / 60.0, np.ones_like(n[mask2]) * 255, np.zeros_like(n[mask2]))).T
    # colors[mask3] = np.vstack((np.ones_like(n[mask3]) * 255, (255 - n[mask3]) * 255 / (255 - 150.0), np.zeros_like(n[mask3]))).T

    colors = np.clip(colors, 0, 255) # 去掉了原代码中的位运算，在最后 np.clip 一下替代
    colors = colors / 255.0 if out_norm else colors.astype(np.uint8)
    return colors
