import numpy as np
import pickle
from pu4c.det3d.common_utils import limit_period, transform_matrix
from . import config


def create_waymo_vis_infos(pkl, valid_classes=config.waymo_kitti_classes[:3], num_features=6, add_ext_info=True):
    infos = pickle.load(open(pkl, 'rb'))

    map_cam_name2id = {'CAM_FRONT': 0, 'CAM_FRONT_LEFT': 1, 'CAM_FRONT_RIGHT': 2, 'CAM_SIDE_LEFT': 3, 'CAM_SIDE_RIGHT': 4}
    map_label2name = {v: k for k, v in infos['metainfo']['categories'].items()}
    spilt = "training" if ("train" in pkl or "val" in pkl) else "testing"
    vis_infos = []
    for info in infos['data_list']:
        cloudpath = f"{spilt}/velodyne/{info['lidar_points']['lidar_path']}"
        lidar_info = {
            'frame_id': info['sample_idx'],
            'filepath': cloudpath,
            'num_features': num_features,
        }

        image_info = {}
        for key, cam in info['images'].items():
            img_ix = f"image_{map_cam_name2id[key]}"
            imagepath = f"{spilt}/{img_ix}/{cam['img_path']}"

            lidar2pixel_mat = np.array(cam['lidar2img'])
            image_info[key] = {
                'filepath': imagepath,
                'l2p_mat': lidar2pixel_mat,
            }

        names, boxes, num_pts = [], [], []
        for instance in info['instances']:
            name = map_label2name[instance['bbox_label_3d']]
            if name in valid_classes:
                # 3D 框需要计算，参考 WaymoDataset.parse_ann_info/CameraInstance3DBoxes.convert_to
                # camera_name = map_cam_id2name[int(instance['camera_id'])] # 用每个相机自己的矩阵反而是错了
                # lidar2cam = np.array(info['images'][camera_name]['lidar2cam'])
                lidar2cam = np.array(info['images']['CAM_FRONT']['lidar2cam'])
                cam2lidar = np.linalg.inv(lidar2cam)

                # box 转坐标系还挺麻烦
                gt_boxes_cam = instance['bbox_3d']
                xyz_ext = np.ones(4)
                xyz_ext[:3] = gt_boxes_cam[:3]
                xyz_ext = xyz_ext @ cam2lidar.T
                # % 这里注意相机坐标系转雷达坐标系，坐标轴要变
                xyz_size = np.array([gt_boxes_cam[3], gt_boxes_cam[5], gt_boxes_cam[4]])
                # 将角度映射 [-pi, pi] 区间内
                yaw = limit_period(-gt_boxes_cam[6] - np.pi / 2, period=np.pi * 2)

                # mmdet3d 中似乎有 bug，没对 z 纠正导致框高度上有偏移
                xyz_ext[2] = xyz_ext[2] + (xyz_size[2] / 2)

                names.append(name)
                boxes.append(np.concatenate([xyz_ext[:3], xyz_size, yaw[..., np.newaxis]]))
                num_pts.append(instance['num_lidar_pts'])
                # difficultys.append(instance['difficulty'])
        annos = {
            'name': np.array(names),
            'gt_boxes_lidar': np.array(boxes),
            'num_points_in_gt': np.array(num_pts),
        }
        vis_infos.append({'lidar': lidar_info, 'image': image_info, 'annos': annos})

    if add_ext_info:
        import os

        key = 'train' if 'train' in pkl else 'val'
        cache = f'/tmp/pu4c/waymo_ext_infos_{key}.pkl'
        if os.path.exists(cache):
            print(f"use cached file: {cache}")
            all_ext_infos = pickle.load(open(cache, 'rb'))
        else:
            from waymo_open_dataset import dataset_pb2
            import tensorflow as tf
            from tqdm import tqdm
            from glob import glob
            key = 'training' if 'train' in pkl else 'validation'
            load_dir = f"{config.waymo_raw_root}/{key}"
            tfrecord_pathnames = sorted(glob(os.path.join(load_dir, '*.tfrecord')))

            def get_ext_info_sigle(sequence_file):
                dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
                ext_infos = []
                for data in dataset:
                    frame = dataset_pb2.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))
                    time_of_day = frame.context.stats.time_of_day.lower()
                    weather = frame.context.stats.weather.lower()
                    location = frame.context.stats.location.lower()
                    ext_info = {
                        'desc': f"{time_of_day} {weather} {location}"
                    }
                    ext_infos.append(ext_info)
                return ext_infos

            num_workers=8
            import concurrent.futures as futures
            with futures.ThreadPoolExecutor(num_workers) as executor:
                ext_infos_list = list(tqdm(executor.map(get_ext_info_sigle, tfrecord_pathnames), total=len(tfrecord_pathnames)))

            all_ext_infos = [ext_info for ext_info_seq in ext_infos_list for ext_info in ext_info_seq]
            with open(cache, 'wb') as f:
                pickle.dump(all_ext_infos, f)

        assert (len(all_ext_infos) == len(vis_infos))
        for i in range(len(all_ext_infos)):
            vis_infos[i]['lidar']['desc'] = all_ext_infos[i]['desc']

    return vis_infos




def draw_hist(data_dict: dict, min, max, num_bins,
              xticks=5, colormap=['blue', 'green', 'red'],
              input_hist=None, normlize=False, save_fname=None,
              title=None, xlabel=None, ylabel=None, figsize=None,
              ):
    """
    Args:
        data_dict: {图例: 直方图数据, ...}
        min, max, num_bins: 截断统计值(x 轴)的范围，以及柱子数量
        xticks: x 轴刻度数
        colormap: 柱子颜色
        normlize: 柱子高度为统计个数还是所占百分比
    值域闭括号 [min, max] 可取最值，xticks 根据 num_bins 的值计算，xticks 的值为真实值，并非均匀选取
    Example:
        hist = draw_hist({'arr': np.array([0,1,2,3,4,5,5,5,8,9])}, 0, 9, 10, xticks=10)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.clf()
    if figsize: plt.figure(figsize=figsize)

    x = np.linspace(min, max, num_bins)
    for i, (key, data) in enumerate(data_dict.items()):
        if input_hist:
            hist = data
        else:
            hist, bins = np.histogram(data, bins=num_bins, range=(min, max))
        if normlize: hist = hist / sum(hist) * 100

        dataframe = pd.DataFrame({'x':x, key:hist})
        sns.barplot(x='x', y=key, data=dataframe, color=colormap[i], alpha=0.5, label=key)

    # 设置 x 轴刻度显示
    show_ticks = xticks  # 设置要显示的刻度数量
    x_ticks = np.linspace(0, x.shape[0] - 1, show_ticks).astype(int)
    x_ticklabels = np.round(x[x_ticks])
    plt.xticks(x_ticks, x_ticklabels)

    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)

    plt.legend()
    if save_fname:
        plt.savefig(save_fname)
    else:
        plt.show()

    return hist