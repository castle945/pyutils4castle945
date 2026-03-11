
def playcloud(switch_func, length, start=0, step=10, point_size=1, background_color=[0, 0, 0], init_camera_params=None):
    """视角参数这块容易引起版本冲突，如无必要可删除，在 open3d-v0.18.0 测试通过"""
    def switch_wrapper(vis, i):
        """保存用户手动调整的视角参数"""
        global camera_params
        # @! Bug: 莫名奇妙的错误，初始化设置视角参数正确，但是到此处读取数据变成 nan 了
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()

        switch_func(vis, i)

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        vis.update_renderer()
    def print_camera_params(vis):
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        rprint({
            'intrinsic': {
                'width': camera_params.intrinsic.width,
                'height': camera_params.intrinsic.height,
                'intrinsic_matrix': camera_params.intrinsic.intrinsic_matrix.tolist(),
            },
            'extrinsic': camera_params.extrinsic.tolist(),
        })
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
    vis.register_key_callback(ord('C'), print_camera_params) # 打印相机参数，Ctrl+C 复制的相机参数还需要一堆计算才能填到 PinholeCameraParameters

    # 初始化相机参数
    if init_camera_params is not None:
        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if 'intrinsic' in init_camera_params:
            # 内参不会变其实不用管
            camera_params.intrinsic.width = init_camera_params['intrinsic']['width']
            camera_params.intrinsic.height = init_camera_params['intrinsic']['height']
            camera_params.intrinsic.intrinsic_matrix = np.array(init_camera_params['intrinsic']['intrinsic_matrix'])
        if 'extrinsic' in init_camera_params:
            camera_params.extrinsic = np.array(init_camera_params['extrinsic'])
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        vis.update_renderer()

    global g_idx
    g_idx = start
    switch_wrapper(vis, start)
    vis.run()
    vis.destroy_window()
