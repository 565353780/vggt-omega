import sys
sys.path.append('../camera-control')
sys.path.append('../colmap-manage')
sys.path.append('../vggsfm-ba')
sys.path.append('../../../camera-control')
sys.path.append('../../../vggsfm-ba')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np

from shutil import rmtree

from camera_control.Method.pcd import toPcd
from camera_control.Module.camera_convertor import CameraConvertor

from colmap_manage.Module.colmap_renderer import COLMAPRenderer

from vggt_omega.Module.vggt_omega_ba_detector import VGGTOmegaBADetector


def demo():
    home = os.environ['HOME']
    work_space = f'{home}/chLi/MMVideoReconV1/'
    model_file_path = f'{home}/chLi/Model/VGGT-Omega/vggt_omega_1b_512.pt'
    vggsfm_model_file_path = home + '/chLi/Model/VGGT/vggsfm_v2_tracker.pt'
    device = 'cuda:0'
    image_resolution = 512
    preprocess_mode = 'balanced'
    test_folder_path = f'{work_space}/lichanghao/20260426_103418_920480/'
    video_file_path = test_folder_path + 'input_video.mov'
    image_folder_path = test_folder_path + 'vggt_omega_ba_test/images/'
    save_colmap_folder_path = test_folder_path + 'vggt_omega_ba_test/'

    vggt_omega_ba_detector = VGGTOmegaBADetector(
        model_file_path=model_file_path,
        vggsfm_model_file_path=vggsfm_model_file_path,
        device=device,
        vggsfm_device=device,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
    )

    output = vggt_omega_ba_detector.detectVideoFile(
        video_file_path=video_file_path,
        save_image_folder_path=image_folder_path,
        target_image_num=72,
    )

    assert output is not None

    camera_list, result = output

    assert camera_list is not None

    print('camera num:', len(camera_list))

    is_ba_success = result['is_ba_success']
    print('is_ba_success:', is_ba_success)
    if is_ba_success:
        pcd = toPcd(result['points_ba'], result['colors_ba'])
    else:
        print('failure_reason:', result['failure_reason'])

        pcd = CameraConvertor.createDepthPcd(
            camera_list=camera_list,
            conf_thresh=0.8,
        )

    shape_points = np.asarray(pcd.points)
    min_pt = shape_points.min(axis=0)
    max_pt = shape_points.max(axis=0)
    center = (min_pt + max_pt) * 0.5
    extent = max_pt - min_pt
    max_extent = max(np.max(extent), 1e-8)
    scale = 4.0 / max_extent

    T = np.eye(4, dtype=np.float64)
    T[0, 0] = T[1, 1] = T[2, 2] = scale
    T[3, 0] = -center[0] * scale
    T[3, 1] = -center[1] * scale
    T[3, 2] = -center[2] * scale

    camera_list = CameraConvertor.transformCameras(camera_list, T)
    # Open3D geometry.transform expects a column-vector/left-multiply matrix.
    pcd.transform(T.T)

    if os.path.exists(save_colmap_folder_path):
        rmtree(save_colmap_folder_path)

    CameraConvertor.createColmapDataFolder(
        cameras=camera_list,
        pcd=pcd,
        save_data_folder_path=save_colmap_folder_path,
        point_num_max=20000,
    )

    COLMAPRenderer.renderColmap(save_colmap_folder_path)
    return True
