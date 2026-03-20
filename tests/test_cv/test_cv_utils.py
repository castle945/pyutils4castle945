import pytest
import pu4c
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy

RPC, RUNVIS = False, False

def test_plot_colormap():
    colormap = pu4c.cv.utils.det25cls_colormap
    pu4c.cv.utils.plot_colormap(colormap, save_path='work_dirs/det25cls_colormap.png')


def test_det3d_utils():
    # 测试框互转角点
    # 定义一个前左上的框（原点处长 3 宽 2 高 1 偏航角 90 度即车头朝向 Y 轴正轴）
    boxes3d = np.array([[0, 0, 0, 3, 2, 1, np.pi/2]])
    corners, lines = pu4c.cv.utils.boxes3d_to_corners_and_lines(boxes3d)
    corners_gt = np.array([
        [ 1., -1.5, -0.5],
        [ 1.,  1.5, -0.5],
        [-1., -1.5, -0.5],
        [ 1., -1.5,  0.5],
        [-1.,  1.5,  0.5],
        [-1., -1.5,  0.5],
        [ 1.,  1.5,  0.5],
        [-1.,  1.5, -0.5],])
    assert np.allclose(corners, corners_gt)


def test_lidar_to_rangeview():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    points, images, calibs, boxes3d, labels = datadb.get('mmdet3d/kitti/000008')

    # HW=64*720 常规的投影，由于点云个数(10 万级别)远大于 HW，大量点投影到同一个像素而丢失
    fov_up, fov_down, height, width = np.radians(2), np.radians(-24.8), 64, 720 # kitti velodyne params
    range_image, point_idx = pu4c.cv.utils.range_projection(
        points[:, :3], height, width, fov=[fov_up, fov_down],
    )
    # range_image, point_idx = pu4c.cv.utils.lidar_to_rangeview(
    #     points[:, :3], height, width, fov=[fov_up, fov_down, -np.pi, np.pi], 
    # )
    # points2 = pu4c.cv.utils.rangeview_to_lidar(range_image, fov=[fov_up, fov_down, -np.pi, np.pi])
    plt.imsave('work_dirs/range_projection.png', range_image)

    # HW=64*4096 以线性拉伸的方式投影，HW 达到 10 万级，仅有少量点丢失
    # 这也说明转 RV 的过程就类似于做体素化，而且此体素化是受透视遮挡影响的，即 RV 图像就是一种柱坐标系体素栅格
    # resolution, height, width = [np.radians(26.8/256), np.radians(360.0/1024)], 256, 1024 # 垂直/水平视场角宽度 除以 height/width
    resolution, height, width = [np.radians(26.8/64), np.radians(360.0/4096)], 64, 4096
    range_image, point_idx, intensity_image = pu4c.cv.utils.lidar_to_rangeview(
        points[:, :4], height, width, resolution=resolution, fov_offset_down=np.radians(-24.8), return_intensity=True
    )
    plt.imsave('work_dirs/lidar_to_rangeview.png', range_image)
    
    # 查找 RV 像素点对应的点云点
    valid_pixels = [[y, x] for y in range(height) for x in range(width) if range_image[y, x] != -1]
    y, x = valid_pixels[666]
    depth, distance = range_image[y, x], np.linalg.norm(points[point_idx[y, x], :3], ord=2, axis=0)
    print(depth, distance, (depth == distance))

    points2 = pu4c.cv.utils.rangeview_to_lidar(
        range_image, resolution=resolution, fov_offset_down=np.radians(-24.8), intensity_image=intensity_image
    )
    print(points.shape[0], points2.shape[0])

    if RUNVIS:
        # 对比查看原始点云和(转 RV 再转回的点云)，常规投影方式大量点丢失，线性拉伸方式只有少量点没对应上丢失
        point_labels = np.concatenate((np.zeros(points.shape[0]), np.ones(points2.shape[0])), axis=0)
        points = np.concatenate((points[:, :3], points2[:, :3]), axis=0)
        pu4c.cv.cloud_viewer(points=points, point_labels=point_labels, point_colormap=[[0.99, 0.99, 0.99], [0.99, 0, 0]], rpc=RPC)
def test_project_points_to_pixels():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data')
    points, images, calibs, boxes3d, labels = datadb.get('mmdet3d/kitti/000008')
    # 边界框投影到图像: 角点投影到图像上、过滤无效框（假设八个像素角点有一半无效则该框无效）、求最小外接矩阵作为二维边界框
    image = copy.deepcopy(images[0])
    corners3d, lines = pu4c.cv.utils.boxes3d_to_corners_and_lines(boxes3d)
    corners2d, _, mask = pu4c.cv.utils.project_points_to_pixels(corners3d, image.shape, transform_mat=calibs['lidar2img'])
    corners2d, mask = corners2d.reshape(-1, 8, 2), mask.reshape(-1, 8)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for box_corners2d, box_mask in zip(corners2d, mask):
        if np.sum(box_mask) <= 4:
            continue
        valid_corners2d = box_corners2d[box_mask]
        x, y = valid_corners2d.astype(int).T
        rect = [min(x), min(y), max(x), max(y)]
        draw.rectangle(rect, outline=(0, 255, 0), width=2)
    pil_image.save('work_dirs/image_with_boxes3d.png')

    # 点云投影到图像
    image = copy.deepcopy(images[0])
    pixels, pixels_depth, mask = pu4c.cv.utils.project_points_to_pixels(points, image_shape=image.shape, transform_mat=calibs['lidar2img'])
    valid_pixels = pixels[mask].astype(int)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for x, y in valid_pixels:
        draw.point((x, y), fill=(255, 0, 0))
        # r = 1
        # draw.ellipse([x-r, y-r, x+r, y+r], fill="red")
    pil_image.save('work_dirs/image_with_points.png')
    
    # 像素投回点
    if RUNVIS:
        mask = np.logical_and(mask, pixels_depth < 80)
        points = pu4c.cv.utils.project_pixels_to_points(pixels=pixels[mask], depth=pixels_depth[mask], transform_mat=np.linalg.inv(calibs['lidar2img']))
        pu4c.cv.cloud_viewer(points=points, rpc=RPC)


if __name__ == '__main__':
    test_project_points_to_pixels()
