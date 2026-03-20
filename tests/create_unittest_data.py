import fire, pu4c, os
import numpy as np
from PIL import Image
# from matplotlib.image import imread # 禁用，对不同格式图片的行为不一致，默认读取为 uint8 [0,255] 但对 png 格式图片会归一化为 float32 [0,1]

datadb = pu4c.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')

def mmdet3d_nuscenes_infos(ann_file='/workspace/codevault/Det3D/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl'):
    pkl_data = pu4c.common.read_pickle(ann_file)
    new_data = {
        'metainfo': pkl_data['metainfo'],
        'data_list': pkl_data['data_list'][1:3],
    }
    datadb.set('mmdet3d/nuscenes_infos', new_data)

def mmdet3d_kitti_000008(data_root='/workspace/codevault/Det3D/mmdetection3d/data/kitti'):
    from mmdet3d.structures.ops.box_np_ops import box_camera_to_lidar
    pkl_data = pu4c.read_pickle(os.path.join(data_root, 'kitti_infos_val.pkl'))
    info = pkl_data['data_list'][5]
    lidar_path = os.path.join(data_root, 'training', 'velodyne', info['lidar_points']['lidar_path'])
    image_path = os.path.join(data_root, 'training', 'image_2', info['images']['CAM2']['img_path'])
    points = pu4c.read_points(lidar_path, num_features=4)
    images = [np.array(Image.open(image_path))]
    calibs = {
        'lidar2img': np.array(info['images']['CAM2']['lidar2img']), # = cam2img @ lidar2cam
        'lidar2cam': np.array(info['images']['CAM2']['lidar2cam']), # = R0_rect @ Tr_velo_to_cam
        'cam2img': np.array(info['images']['CAM2']['cam2img']),
    }
    gt_boxes_camera = np.array([np.array(obj['bbox_3d']) for obj in info['instances'] if obj['bbox_label_3d'] != -1]) # 去掉 DontCare 目标
    gt_boxes_lidar = box_camera_to_lidar(gt_boxes_camera, np.array(info['images']['R0_rect']), np.array(info['lidar_points']['Tr_velo_to_cam']))
    gt_boxes_lidar[:, 2] += (gt_boxes_lidar[:, 5] / 2)
    boxes3d = gt_boxes_lidar
    labels = np.array([obj['bbox_label_3d'] for obj in info['instances'] if obj['bbox_label_3d'] != -1])
    datadb.set('mmdet3d/kitti/000008', [points, images, calibs, boxes3d, labels])

def semantickitti_000000():
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
    learning_map, classes, colormap = semantickitti_learning_map, semantickitti_classes, semantickitti_colormap
    lidar_path = '/datasets/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin'
    label_path = '/datasets/SemanticKITTI/dataset/sequences/00/labels/000000.label'
    points = pu4c.cv.read_points(lidar_path, num_features=4)
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels & 0xFFFF    # 等价于 % 2**16，取模后最大值为 255
    labels = np.vectorize(learning_map.__getitem__)(labels) # 映射完最大值 19
    labels = labels.astype(np.uint8)
    datadb.set("semantickitti/000000", data=[points, labels, classes, colormap])

def occ3d_nuscenes_scene0001_1e19d0():
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
    classes, colormap = occ3d_classes, occ3d_colormap
    label_path = "/datasets/nuScenes/Occ3D-nuScenes/gts/scene-0001/1e19d0a5189b46f4b62aa47508f2983e/labels.npz"
    data = np.load(label_path)
    labels, lidar_mask, camera_mask = data['semantics'], data['mask_lidar'], data['mask_camera']
    voxel_size, point_cloud_range = [0.4, 0.4, 0.4], np.array([-40, -40, -1, 40, 40, 5.4])
    
    FREE_LABEL = 17                     # 空体素类别
    valid_mask = labels != FREE_LABEL   # 非空体素掩膜，HWD
    valid_idx = np.where(valid_mask)    # 非空体素下标，tuple(X, Y, Z), X.shape(N=非空体素个数,)
    labels = labels[valid_idx]          # 非空体素的标签，(N,)，值域 0-16 共 17 类
    voxel_centers = np.array(valid_idx).T * voxel_size # 非空体素坐标即中心点坐标，(N, 3)，等于下标乘以体素大小
    voxel_centers = voxel_centers - ((point_cloud_range[3:6] - point_cloud_range[0:3]) / 2) # 坐标原点移动到场景中心

    colormap = np.array(colormap) / 255 # 归一化颜色值
    datadb.set("occ3d_nuscenes/scene0001-1e19d0", data=[voxel_centers, voxel_size, labels, classes, colormap])


if __name__ == '__main__':
    fire.Fire()
