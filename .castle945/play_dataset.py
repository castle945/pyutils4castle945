
semantickitti_learning_map = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
semantickitti_classes = [
    'unlabeled',
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation',
    'trunck', 'terrian', 'pole', 'traffic-sign'
]
semantickitti_colormap = [
    [255, 255, 255],
    [100, 150, 245], [100, 230, 245], [30, 60, 150],
    [80, 30, 180], [100, 80, 250], [155, 30, 30],
    [255, 40, 200], [150, 30, 90], [255, 0, 255],
    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0],
    [255, 120, 50], [0, 175, 0], [135, 60, 0], [150, 240, 80],
    [255, 240, 150], [255, 0, 0]
]

# copy from https://github.com/Tsinghua-MARS-Lab/CVT-Occ/blob/8291052c9bdb402e4706bed842e3ec9fd28e0bf5/projects/configs/cvtocc/cvtocc_nuscenes.py
occ3d_classes = [
    'others',               # 也即被占用未知类别的通用物体 GO
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',    # 路面
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
]
# 红橙色系(车): 吊车红色、拖车洋红、卡车暗红、公交车粉红、汽车橙色
# 黄色系(小型交通参与者): 行人黄色、摩托车葱黄、自行车棕黄
# 绿色: 树
# 蓝色系(静态障碍): 围挡障碍蓝色、交通锥靛蓝
# 靛紫色系(路面): 路面靛色、人行道浅浅紫、地形浅紫
# 黑白色系(墙面隔断): 墙面浅灰、道中屏障(常见于双向车道高速路)深灰、通用障碍黑色
occ3d_colormap = [
    [0  , 0  , 0  ],  # 0  others               通用障碍     黑色
    [0  , 0  , 255],  # 1  barrier              围挡障碍     蓝色
    [174, 112, 0  ],  # 2  bicycle              自行车      棕黄
    [255, 180, 170],  # 3  bus                  公交车      粉红
    [255, 128, 0  ],  # 4  car                  汽车        橙色
    [255, 0  , 0  ],  # 5  construction_vehicle 建筑车辆吊车  红色
    [163, 217, 0  ],  # 6  motorcycle           摩托车      葱黄
    [255, 255, 0  ],  # 7  pedestrian           行人        黄色
    [6  , 82 , 121],  # 8  traffic_cone         交通锥      靛蓝
    [255, 0  , 150],  # 9  trailer              拖车        洋红
    [160, 0  , 0  ],  # 10 trunk                卡车        暗红
    [0  , 255, 255],  # 11 driveable_surface    路面        靛色
    [100, 100, 100],  # 12 other_flat           道中屏障     深灰
    [255, 180, 255],  # 13 sidewalk             人行道       浅浅紫
    [255, 100, 255],  # 14 terrain    地形即人行道外的路面     浅紫
    [200, 200, 200],  # 15 manmade         人造建筑如墙等     浅灰
    [0  , 180, 0  ],  # 16 vegetation           植被        绿色
]

# 数据集可视化
@rpc_func
def play_semantickitti(
    root: str = '/datasets/SemanticKITTI',
    start: int = 0, step: int = 10,
    rpc: bool = False,
) -> None:
    """播放 SemanticKITTI 数据集"""
    import glob, os
    import numpy as np
    from .utils import open3d_utils

    assert os.path.exists(root)
    sequencepaths = [os.path.join(root, 'dataset/sequences', str(i).zfill(2)) for i in range(11)] # 只有 seq0-10 是有标签的
    filepaths = sorted(glob.glob(f'{sequencepaths[0]}/labels/*.label'))
    global next_seq
    next_seq = 1
    colormap = np.array(cfg.semantickitti_colormap) / 255

    def switch(vis, i):
        if i >= len(filepaths):
            global next_seq
            print(f'loading frames from {sequencepaths[next_seq]}...')
            filepaths.extend(sorted(glob.glob(f'{sequencepaths[next_seq]}/labels/*.label')))
            next_seq += 1
        label_path = filepaths[i]
        print(f"frame {i}: {label_path}")
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels & 0xFFFF    # 等价于 % 2**16，取模后最大值为 255
        labels = np.vectorize(cfg.semantickitti_learning_map.__getitem__)(labels) # 映射完最大值 19
        # 如果要将 [0, 1-19] 的标签移动为 [0-18, 255] 则解注释，但那样未标记的点标签 255 不好写 colormap
        # ignore_label = 255          # 任意的一个大值，与有标记的语义类别区分开就行
        # labels[labels == 0] = ignore_label + 1  # 映射后的标签 0-19，其中 0 为未标记的点，将未标记的点赋值为 ignore_label，由于之后所有标签要 -1 故这里要赋值为 ignore_label+1
        # labels = labels - 1                     # 经上一行将未标记的点标签重新赋值后，此时 1-19 为有标记的点，将其标签整体 -1 则 0-18 为最终的语义标签，255 为未标记
        labels = labels.astype(np.uint8)
        cloud_viewer(
            filepath=label_path.replace('labels', 'velodyne')[:-5] + 'bin', num_features=4,
            point_labels=labels, cloud_colormap=colormap,
            vis=vis, show_axis=True, run=False, 
            )
    
    open3d_utils.playcloud(switch, length=18660, start=start, step=step)

@rpc_func
def play_occ3d_nuscenes(
    root: str = '/datasets/nuScenes/Occ3D-nuScenes',
    start: int = 0, step: int = 10,
    rpc: bool = False,
) -> None:
    """播放 Occ3D-nuScenes 数据集

    labels.npz 整个文件类似于一个字典，包括
        semantics: (H=80/0.4=200, W=80/0.4=200, D=6.4/0.4=16), 每个体素的语义标签
        mask_lidar: HWD, 激光雷达视角下的掩膜
        mask_camera: HWD, 相机视角下的掩膜
    """
    import glob, os
    import numpy as np
    from .utils import open3d_utils

    assert os.path.exists(root)
    # 如果数据集是以 nfs 挂载到本地计算机，一次性搜索所有文件太慢，动态加载
    # filepaths = sorted(glob.glob(f"{os.path.join(root, 'gts')}/scene-*/*/labels.npz"))
    scenepaths = sorted(glob.glob(f"{os.path.join(root, 'gts')}/scene-*"))
    filepaths = sorted(glob.glob(f'{scenepaths[0]}/*/labels.npz'))
    global next_scene
    next_scene = 1
    colormap = np.array(cfg.occ3d_colormap) / 255

    def switch(vis, i):
        if i >= len(filepaths):
            global next_scene
            print(f'loading frames from {scenepaths[next_scene]}...')
            filepaths.extend(sorted(glob.glob(f'{scenepaths[next_scene]}/*/labels.npz')))
            next_scene += 1
        label_path = filepaths[i]
        print(f"frame {i}: {label_path}")
        data = np.load(label_path)
        labels, lidar_mask, camera_mask = data['semantics'], data['mask_lidar'], data['mask_camera']
        voxel_size, point_cloud_range = [0.4, 0.4, 0.4], np.array([-40, -40, -1, 40, 40, 5.4])
        
        FREE_LABEL = 17                     # 空体素类别
        valid_mask = labels != FREE_LABEL   # 非空体素掩膜，HWD
        valid_idx = np.where(valid_mask)    # 非空体素下标，tuple(X, Y, Z), X.shape(N=非空体素个数,)
        labels = labels[valid_idx]          # 非空体素的标签，(N,)，值域 0-16 共 17 类
        voxel_centers = np.ascontiguousarray(np.array(valid_idx).T) * voxel_size + voxel_size * 0.5 # 非空体素坐标即中心点坐标，(N, 3)，等于下标乘以体素大小
        voxel_centers = voxel_centers - ((point_cloud_range[3:6] - point_cloud_range[0:3]) / 2) # 坐标原点移动到场景中心

        voxel_viewer(
            voxel_centers, voxel_size,
            voxel_labels=labels, voxel_colormap=colormap,
            vis=vis, run=False, 
            )
    
    open3d_utils.playcloud(switch, length=34149, start=start, step=step, point_size=10, background_color=[1, 1, 1])
@rpc_func
def play_surroundocc(
    root: str = '/datasets/nuScenes/surroundocc',
    start: int = 0, step: int = 10,
    rpc: bool = False,
) -> None:
    """播放 surroundocc 数据集"""
    import glob, os
    import numpy as np
    from .utils import open3d_utils

    assert os.path.exists(root)
    filepaths = sorted(glob.glob(f"{os.path.join(root, 'samples')}/*.npy"))
    colormap = np.array(cfg.occ3d_colormap) / 255

    def switch(vis, i):
        label_path = filepaths[i]
        print(f"frame {i}: {label_path}")
        sparse_labels = np.load(label_path)
        voxel_size, point_cloud_range = [0.5, 0.5, 0.5], np.array([-50, -50, -5, 50, 50, 3])

        voxel_centers = np.vstack((sparse_labels[:, 0], sparse_labels[:, 1], sparse_labels[:, 2])).T * voxel_size # 非空体素下标乘以体素大小得到非空体素坐标
        voxel_centers = voxel_centers + point_cloud_range[0:3] # 坐标原点移动到场景中心
        labels = sparse_labels[:, 3]

        voxel_viewer(
            voxel_centers, voxel_size,
            voxel_labels=labels, voxel_colormap=colormap,
            vis=vis, run=False, 
            )
    
    open3d_utils.playcloud(switch, length=len(filepaths), start=start, step=step, point_size=10, background_color=[1, 1, 1])

