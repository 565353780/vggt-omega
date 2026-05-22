import sys
sys.path.append('../camera-control')
sys.path.append('../colmap-manage')
sys.path.append('../../../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from shutil import rmtree

from camera_control.Module.camera_convertor import CameraConvertor

from colmap_manage.Module.colmap_renderer import COLMAPRenderer

from vggt_omega.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    work_space = f'{home}/chLi/MMVideoReconV1/'
    model_file_path = f'{home}/chLi/Model/VGGT-Omega/vggt_omega_1b_512.pt'
    device = 'cuda:0'
    image_resolution = 512
    preprocess_mode = 'balanced'
    test_folder_path = f'{work_space}/lichanghao/20260426_103418_920480/'
    video_file_path = test_folder_path + 'input_video.mov'
    image_folder_path = test_folder_path + 'vggt_omega_test/images/'
    save_folder_path = test_folder_path + 'vggt_omega_test/'

    detector = Detector(
        model_file_path,
        device,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
    )

    camera_list = detector.detectVideoFile(
        video_file_path=video_file_path,
        save_image_folder_path=image_folder_path,
        target_image_num=200,
    )

    assert camera_list is not None

    print('camera num:', len(camera_list))

    pcd = CameraConvertor.createDepthPcd(
        camera_list=camera_list,
        conf_thresh=0.0,
    )

    save_colmap_folder_path = save_folder_path + 'colmap/'
    if os.path.exists(save_colmap_folder_path):
        rmtree(save_colmap_folder_path)

    CameraConvertor.createColmapDataFolder(
        cameras=camera_list,
        pcd=pcd,
        save_data_folder_path=save_colmap_folder_path,
        point_num_max=200000,
    )

    COLMAPRenderer.renderColmap(save_colmap_folder_path)
    return True
