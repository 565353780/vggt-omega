import os

from typing import List, Optional

from camera_control.Module.camera import Camera

from vggt_omega.Module.detector import Detector


home = os.environ['HOME']
vggt_omega_model_file_path = f'{home}/chLi/Model/VGGT-Omega/vggt_omega_1b_512.pt'


def get_default_model_paths() -> dict:
    """Return the default VGGT-Omega detector model paths.

    与 ``flux_mv.API.sampler.get_default_model_paths`` 风格保持一致，便于上层
    ``video_pipeline.queryModelPaths()`` 统一聚合所有模块的模型路径。
    """
    return {
        'model_file_path': vggt_omega_model_file_path,
    }


def build_detector(
    model_file_path: str = vggt_omega_model_file_path,
    device: str = 'cuda:0',
    image_resolution: int = 512,
    preprocess_mode: str = 'balanced',
    is_offload_cpu: bool = True,
) -> Detector:
    """Build a VGGT-Omega :class:`Detector`.

    传参与 ``pixel-align-deform/video_pipeline.py`` 中的 ``VGGTOmegaDetector``
    构造保持一致。
    """
    return Detector(
        model_file_path=model_file_path,
        device=device,
        image_resolution=image_resolution,
        preprocess_mode=preprocess_mode,
        is_offload_cpu=is_offload_cpu,
    )


def detect_image_folder(
    detector: Detector,
    image_folder_path: str,
) -> Optional[List[Camera]]:
    """Run VGGT-Omega inference over an image folder and return cameras."""
    return detector.detectImageFolder(
        image_folder_path=image_folder_path,
    )
