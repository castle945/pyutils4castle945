from typing import Any, Union, Optional, Dict, List, Tuple, Callable
import open3d as o3d
import numpy as np
from . import rviz_intensity_colormap, boxes3d_to_corners_and_lines

def create_pointcloud_geometry(
    points: np.ndarray,
    labels: np.ndarray = None,
    colormap: Union[List[List], np.ndarray] = None,
):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    if labels is not None:
        assert colormap is not None
        colors = np.array(colormap)[labels.astype(np.int64)]
        cloud.colors = o3d.utility.Vector3dVector(colors)
    elif isinstance(colormap, list) and len(colormap) == 1:
        # 若颜色映射仅一种颜色则着此纯色，即 [[r,g,b]]
        cloud.paint_uniform_color(colormap[0])
    elif points.shape[1] == 4:
        colors = rviz_intensity_colormap(points[:, 3], out_norm=True)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return cloud
def create_boxes3d_geometry(
    boxes3d: np.ndarray,
    colormap: Union[List[List], np.ndarray] = None,
    show_heading: bool = True,
):
    N, C = boxes3d.shape
    corners, lines = boxes3d_to_corners_and_lines(boxes3d[:, :7], show_heading=show_heading)
    line_sets = o3d.geometry.LineSet()
    line_sets.points = o3d.open3d.utility.Vector3dVector(corners)
    line_sets.lines = o3d.open3d.utility.Vector2iVector(lines)
    if C > 7:
        assert np.max(boxes3d[:, 7]) < len(colormap), 'Number of classes is bigger than number of colors'
        box3d_colors = np.array(colormap)[boxes3d[:, 7].astype(np.int64)]
        colors = np.repeat(box3d_colors, 14 if show_heading else 12, axis=0) # (N, 3) -> (N*14 or N*12, 3)
        line_sets.colors = o3d.utility.Vector3dVector(colors)
    elif isinstance(colormap, list) and len(colormap) == 1:
        # 若颜色映射仅一种颜色则着此纯色，即 [[r,g,b]]
        line_sets.paint_uniform_color(colormap[0])
    else:
        line_sets.paint_uniform_color([0.0, 1.0, 0.0])

    return line_sets
def create_voxels_geometry(voxel_centers: np.ndarray, voxel_size: np.ndarray, borders_color: List[float] = [[0, 0, 0]]):
    """批量添加边框，可视化体素就是每个中心点都画一个框"""
    # 计算每个非空体素的边框 (N,7)[xyz,wlh,yaw]
    N = voxel_centers.shape[0]
    wlh = np.tile(voxel_size, (N, 1)) # voxel_size 的第 0 维重复 N 次，第 1 维重复 1 次
    yaw = np.zeros((N, 1))
    boxes3d = np.concatenate((voxel_centers, wlh, yaw), axis=1)

    return create_boxes3d_geometry(boxes3d, colormap=borders_color, show_heading=False)

def playcloud(
    switch_func: Callable, 
    length: int,
    start: int = 0,
    step: int = 10,
    point_size: int = 1,
    background_color: List[float] = [0, 0, 0],
):
    """视角参数这块容易引起版本冲突，如无必要可删除，在 open3d-v0.18.0 测试通过"""
    def switch_wrapper(vis, i):
        """保存用户手动调整的视角参数"""
        global camera_params
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        switch_func(vis, i)

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        vis.update_renderer()
    def prev(vis):
        global g_idx
        g_idx = max(g_idx - 1, 0)
        switch_wrapper(vis, g_idx)
    def next(vis):
        global g_idx
        g_idx = min(g_idx + 1, length-1)
        switch_wrapper(vis, g_idx)
    def prev_n(vis):
        global g_idx
        g_idx = max(g_idx - step, 0)
        switch_wrapper(vis, g_idx)
    def next_n(vis):
        global g_idx
        g_idx = min(g_idx + step, length-1)
        switch_wrapper(vis, g_idx)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = background_color

    vis.register_key_callback(ord('W'), prev_n)
    vis.register_key_callback(ord('S'), next_n)
    vis.register_key_callback(ord('A'), prev)
    vis.register_key_callback(ord('D'), next) # 按小写，但这里要填大写

    global g_idx
    g_idx = start
    switch_wrapper(vis, start)
    vis.run()
    vis.destroy_window()