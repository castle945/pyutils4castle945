from typing import Any, Union, Optional, Dict, List, Tuple
from pu4c.common.utils import rpc_func
from .utils import det25cls_colormap
import numpy as np

@rpc_func
def cloud_viewer(
    # 点云来源，直接传入或读文件
    points: np.ndarray = None,
    filepath: str = None, 
    num_features: int = 4,
    # 边界框，用于三维目标检测可视化 
    boxes3d: np.ndarray = None,
    boxes3d_colormap: Union[List[List], np.ndarray] = det25cls_colormap,
    # 点标签及点标签颜色映射，用于三维语义分割可视化
    point_labels: np.ndarray = None, 
    point_colormap: Union[List[List], np.ndarray] = None,
    # 杂项
    vis: Any = None,
    show_axis: bool = True,
    runvis: bool = True,
    rpc: bool = False,
) -> None:
    """快速查看单帧点云，支持 pcd/bin/npy/pkl/txt
    Args:
        points (ndarray(N, 3) | ndarray(N, 4)[x,y,z,i])
        boxes3d (ndarray(N, 7)[xyz,lwh,yaw] | ndarray(N, 8)[xyz,lwh,yaw,cls])
        point_colormap (list[list[r,g,b]] | ndarray(M, 3)): 点标签颜色映射，用于将整型的点标签映射为 [r,g,b] 值，open3d 要求颜色值已归一化
        show_axis: 是否绘制坐标轴，如果不绘制则会自动调整观察视角（点云偏离原点很远时有用）
    Examples:
        pu4c.cv.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000000.bin", num_features=4)
        pu4c.cv.cloud_viewer(points, boxes3d=boxes3d, rpc=True)
        pu4c.cv.cloud_viewer(points, point_labels=point_labels, point_colormap=colormap)
    Keys:
        -/=: 调整点云点的大小
    Notes:
        - 关于点云着色: 优先级为   根据点标签颜色映射着色（若有点标签输入则按标签着色，否则若颜色映射仅一种颜色则着此纯色）> 反射率着色（若输入点云带反射率属性）> 高度着色（若既无颜色映射也无反射率，则 open3d 默认按照高度着色）
        - 关于边界框着色: 优先级为 根据框标签颜色映射着色（若有框标签输入即 (N, 8) 则按标签着色，否则若颜色映射仅一种颜色则着此纯色）> 着绿色
        - 关于边界框坐标系/物体坐标系: 要求输入的边界框物体坐标系为前左上，即 X 轴正向处偏航角为 0
                                    up z    x front (yaw=0)
                                        ^   ^
                                        |  /
                                        | /
            (yaw=0.5*pi) left y <------ 0
        - 注意事项: 颜色映射中的颜色值必须归一化，虽然不归一化并不报错，但是只能正确显示部分颜色导致与预期相悖
    """
    import open3d as o3d
    from .utils import read_points, open3d_utils
    
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1
        vis.get_render_option().background_color = np.zeros(3)
    if show_axis:
        axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_geometry)

    if filepath is not None:
        points = read_points(filepath, num_features=num_features)
    if points is not None:
        cloud_geometry = open3d_utils.create_pointcloud_geometry(
            points, labels=point_labels, colormap=point_colormap,
        )
        vis.add_geometry(cloud_geometry)
    if boxes3d is not None:
        boxes3d_geometry = open3d_utils.create_boxes3d_geometry(boxes3d, colormap=boxes3d_colormap)
        vis.add_geometry(boxes3d_geometry)

    if runvis:
        vis.run()
        vis.destroy_window()
@rpc_func
def voxel_viewer(
    # 体素中心点坐标及体素大小
    voxel_centers: np.ndarray,
    voxel_size: np.ndarray,
    # 体素标签及体素标签颜色映射，用于三维语义占用预测可视化
    voxel_labels: np.ndarray = None,
    voxel_colormap: Union[List[List], np.ndarray] = None,
    # 杂项
    vis: Any = None,
    show_axis: bool = True,
    runvis: bool = True,
    rpc: bool = False,
) -> None:
    """体素可视化
    Args:
        voxel_centers (ndarray(N, 3))
        voxel_size (ndarray(3,))
        voxel_labels (ndarray(N,))
        voxel_colormap (list[list[r,g,b]] | ndarray(M, 3))
    Examples:
        pu4c.cv.voxel_viewer(voxel_centers, voxel_size, voxel_labels=labels, voxel_colormap=colormap)
    Notes:
        - 关于体素着色: 中心点着色 优先级为 根据体素标签颜色映射着色（若有体素标签输入则按标签着色，否则若颜色映射仅一种颜色则着此纯色）> 着绿色，边框默认着黑色
    """
    import open3d as o3d
    from .utils import open3d_utils
    
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 10
        vis.get_render_option().background_color = [1, 1, 1]
    if voxel_centers is not None:
        voxels_geometry = open3d_utils.create_voxels_geometry(voxel_centers, voxel_size)
        vis.add_geometry(voxels_geometry)

    # 如有标签优先按标签着色，否则着纯色
    cloud_viewer(points=voxel_centers, point_labels=voxel_labels,
        point_colormap=voxel_colormap if voxel_colormap is not None else [[0, 1, 0]],
        vis=vis, show_axis=show_axis, runvis=runvis,
    )
@rpc_func
def cloud_viewer_panels(
    points_list: List[np.ndarray],
    boxes3d_list: List[np.ndarray] = None,
    point_labels_list: List[np.ndarray] = None,
    point_colormap: Union[List[List], np.ndarray] = None,
    offset: List[float] = None,
    show_axis: bool = True,
    runvis: bool = True,
    rpc: bool = False,
) -> None:
    """同一个窗口中可视化多个点云，共享视角参数
    Args:
        offset: 面板之间的间隔，open3d 窗口坐标系，右前上
    Examples:
        pu4c.cv.cloud_viewer_panels(points_list=[points1, points2], boxes3d_list=[boxes3d1, boxes3d2], offset=[180, 0, 0])  
    """
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.zeros(3)
    if show_axis:
        axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_geometry)

    offset = np.array([100, 0, 0]) if offset is None else np.array(offset)
    for i in range(len(points_list)):
        points = points_list[i][:, :3] + (offset * i)
        point_labels = point_labels_list[i] if point_labels_list is not None else None
        boxes3d = boxes3d_list[i] if boxes3d_list is not None else None
        if boxes3d_list is not None:
            boxes3d[:, :3] += (offset * i)
        cloud_viewer(points=points, point_labels=point_labels, boxes3d=boxes3d,
            point_colormap=point_colormap, vis=vis, show_axis=False, runvis=False, 
        )

    if runvis:
        vis.run()
        vis.destroy_window()
@rpc_func
def cloud_player(
    root: str = None,
    pattern: str = '*',
    filepaths: List[str] = None,
    num_features: int = 4,
    points_list: List[np.ndarray] = None,
    boxes3d_list: List[np.ndarray] = None,
    show_axis: bool = True,
    start: int = 0,
    step: int = 10,
    rpc: bool = False,
) -> None:
    """点云播放器，支持播放点云目录与点云列表
    - 注意设置保存视角参数后，初始视角不再良好，需要滚动鼠标滚轮缩放一下才能看到
    Examples:
        pu4c.cv.cloud_player(root="/datasets/KITTI/object/training/velodyne/", num_features=4, pattern="*.bin")  
    Keys:
        A/D: pre/next one frame
        W/S: pre/next step frame
        Ctrl+C/V: 复制/粘贴视角参数
    """
    from glob import glob
    from .utils import open3d_utils

    assert (root is not None) or (filepaths is not None) or (points_list is not None)
    if root is not None:
        filepaths = sorted(glob(f'{root}/{pattern}'))
    length = len(points_list) if filepaths is None else len(filepaths)

    def switch(vis, i):
        print_msg = f"frame {i}" if root is None else f"frame {i}: {filepaths[i]}"
        print(print_msg)
        cloud_viewer(
            filepath=None if filepaths is None else filepaths[i],
            points=None if points_list is None else points_list[i],
            boxes3d=None if boxes3d_list is None else boxes3d_list[i],
            num_features=num_features,
            vis=vis, show_axis=show_axis, runvis=False,
        )
    
    open3d_utils.playcloud(switch, length, start=start, step=step)

@rpc_func
def image_viewer(data: np.ndarray = None, filepath: str = None, rpc: bool = False):
    """可视化图像
    Args:
        data (ndarray(H, W, C)): 图片数据，rgb uint8 [0, 255]
    Notes:
        - cv2 可视化更方便但连续远程调用会卡死
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    assert data is not None or filepath is not None
    if data is None:
        data = np.array(Image.open(filepath))
    height, width = data.shape[:2]
    plt.figure(figsize=(width, height), dpi=1)
    plt.imshow(data)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
