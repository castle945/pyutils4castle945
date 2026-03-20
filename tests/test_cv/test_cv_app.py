import pytest
import pu4c
import numpy as np

# 本地有图形界面时，本地执行且不显示交互窗口
RPC, RUNVIS = False, False
# 本地无图形界面时，远程执行且不显示交互窗口
# RPC, RUNVIS = True, False

def test_viewer():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    
    # 三维目标检测可视化
    points, images, calibs, boxes3d, labels = datadb.get('mmdet3d/kitti/000008')
    pu4c.cv.cloud_viewer(points, point_colormap=[[0.99,0.99,0.99]], runvis=RUNVIS, rpc=RPC)
    boxes3d_with_label = np.concatenate((boxes3d, labels[:, None]), axis=1)
    pu4c.cv.cloud_viewer(points, boxes3d=boxes3d_with_label, runvis=RUNVIS, rpc=RPC)
    pu4c.cv.cloud_viewer_panels(points_list=[points, points], boxes3d_list=[boxes3d_with_label, boxes3d_with_label], offset=[180, 0, 0], runvis=RUNVIS, rpc=RPC)

    # 三维语义分割可视化
    points, labels, classes, colormap = datadb.get("semantickitti/000000")
    pu4c.cv.cloud_viewer(points, point_labels=labels, point_colormap=colormap, runvis=RUNVIS, rpc=RPC)

    # # 三维占据预测可视化
    voxel_centers, voxel_size, labels, classes, colormap = datadb.get("occ3d_nuscenes/scene0001-1e19d0")
    pu4c.cv.voxel_viewer(voxel_centers, voxel_size, voxel_labels=labels, voxel_colormap=colormap, runvis=RUNVIS, rpc=RPC)

if __name__ == '__main__':
    test_viewer()
