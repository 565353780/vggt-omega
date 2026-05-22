from typing import Optional, List, Tuple, Dict

from camera_control.Module.camera import Camera

from vggsfm_ba.Module.vggsfm_detector import VGGSfMDetector

from vggt_omega.Module.detector import Detector


class VGGTOmegaBADetector(object):
    '''VGGTOmegaBADetector 是 VGGT-Omega ``Detector`` + ``VGGSfMDetector`` 的轻量编排器：

      1. 用 ``Detector`` 跑 VGGT-Omega 前馈，得到带初始位姿/深度/原图的 ``camera_list``；
      2. 用 ``VGGSfMDetector`` 对 ``camera_list`` 做 VGGSfM + pycolmap BA 联合优化；
      3. BA 成功时使用 BA 重建出来的稀疏点云，失败时回退到深度反投点云；

    本类不再持有任何 VGGT/VGGSfM 推理实现，所有推理细节都委托给上述两个子模块。
    '''

    def __init__(
        self,
        model_file_path: Optional[str] = None,
        vggsfm_model_file_path: Optional[str] = None,
        device: str = 'cuda:0',
        vggsfm_device: str = 'cuda:0',
        image_resolution: int = 512,
        preprocess_mode: str = 'balanced',
        patch_size: int = 16,
        enable_alignment: bool = False,
    ) -> None:
        # 这里先把权重路径置为 None，避免子构造器内部就开始尝试加载；
        # 后面再统一由本类 loadModel / loadVGGSfMModel 委托加载，便于复用。
        self.detector = Detector(
            model_file_path=None,
            device=device,
            image_resolution=image_resolution,
            preprocess_mode=preprocess_mode,
            patch_size=patch_size,
            enable_alignment=enable_alignment,
        )
        self.vggsfm_detector = VGGSfMDetector(
            vggsfm_model_file_path=None,
            device=vggsfm_device,
        )

        if model_file_path is not None:
            self.loadModel(model_file_path, device)
        if vggsfm_model_file_path is not None:
            self.loadVGGSfMModel(vggsfm_model_file_path, vggsfm_device)
        return

    @property
    def device(self) -> str:
        return self.detector.device

    @property
    def vggsfm_device(self) -> str:
        return self.vggsfm_detector.device

    def loadModel(
        self,
        model_file_path: str,
        device: str = 'cuda:0',
    ) -> bool:
        return self.detector.loadModel(model_file_path, device)

    def loadVGGSfMModel(
        self,
        vggsfm_model_file_path: str,
        vggsfm_device: str = 'cuda:0',
    ) -> bool:
        return self.vggsfm_detector.loadModel(vggsfm_model_file_path, vggsfm_device)

    def detectImageFolder(
        self,
        image_folder_path: str,
        **vggsfm_kwargs,
    ) -> Optional[Tuple[List[Camera], Dict]]:
        '''读取 ``image_folder_path`` 下的图像，跑 VGGT-Omega + VGGSfM BA。'''
        camera_list = self.detector.detectImageFolder(
            image_folder_path=image_folder_path,
        )
        if camera_list is None or len(camera_list) == 0:
            print('[ERROR][VGGTOmegaBADetector::detectImageFolder]')
            print('\t Detector.detectImageFolder returned empty camera_list')
            print('\t image_folder_path:', image_folder_path)
            return None

        refined_cameras, result = self.vggsfm_detector.refineCameras(
            camera_list,
            **vggsfm_kwargs,
        )

        return refined_cameras, result

    def detectVideoFile(
        self,
        video_file_path: str,
        save_image_folder_path: str,
        target_image_num: int = 200,
        **vggsfm_kwargs,
    ) -> Optional[Tuple[List[Camera], Dict]]:
        '''先抽帧到 ``save_image_folder_path``，再走 VGGT-Omega + VGGSfM BA。'''
        camera_list = self.detector.detectVideoFile(
            video_file_path=video_file_path,
            save_image_folder_path=save_image_folder_path,
            target_image_num=target_image_num,
        )

        if camera_list is None or len(camera_list) == 0:
            print('[ERROR][VGGTOmegaBADetector::detectVideoFile]')
            print('\t Detector.detectVideoFile returned empty camera_list')
            print('\t video_file_path:', video_file_path)
            return None

        return self.vggsfm_detector.refineCameras(
            camera_list,
            **vggsfm_kwargs,
        )
