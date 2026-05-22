import os
import gc
import warnings

import torch
import numpy as np

from PIL import Image
from shutil import rmtree
from torchvision import transforms as TF
from typing import Optional, List, Tuple, Union, Dict, Any

from camera_control.Module.camera import Camera
from camera_control.Module.camera_convertor import CameraConvertor

from colmap_manage.Method.video import videoToImages

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera


class Detector(object):
    '''Detector 只负责 VGGT-Omega 前馈推理 + 构建 ``camera_list``。

    与旧 VGGT 不同，VGGT-Omega 接受动态分辨率输入：图像按官方
    ``balanced`` 或 ``max_size`` 策略缩放到 ``patch_size`` 的整数倍，
    batch 内不同尺寸会 padding 到共同 ``H,W`` 后再前馈。本类输出的每个
    ``Camera`` 持有该图自身可见区域（去掉 batch padding）的原图分辨率
    RGB / 内参，以及裁回到可见区域的 depth/conf。
    '''

    def __init__(
        self,
        model_file_path: Optional[str]=None,
        device: str = 'cuda:0',
        image_resolution: int = 512,
        preprocess_mode: str = 'balanced',
        patch_size: int = 16,
        enable_alignment: bool = False,
    ) -> None:
        self.device = device
        self.image_resolution = int(image_resolution)
        self.preprocess_mode = str(preprocess_mode)
        self.patch_size = int(patch_size)
        self.enable_alignment = bool(enable_alignment)

        if self.preprocess_mode not in ('balanced', 'max_size'):
            raise ValueError(
                f"[Detector] preprocess_mode must be 'balanced' or 'max_size', got {self.preprocess_mode!r}"
            )
        if self.image_resolution <= 0:
            raise ValueError("[Detector] image_resolution must be positive")
        if self.patch_size <= 0:
            raise ValueError("[Detector] patch_size must be positive")
        if self.image_resolution % self.patch_size != 0:
            raise ValueError(
                "[Detector] image_resolution must be divisible by patch_size"
            )

        self.model = VGGTOmega(
            patch_size=self.patch_size,
            enable_alignment=self.enable_alignment,
        )

        if model_file_path is not None:
            self.loadModel(model_file_path, self.device)
        return

    def loadModel(
        self,
        model_file_path: str,
        device: str = 'cuda:0',
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        self.device = device

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        # 权重加载完始终保留在 CPU，仅在推理窗口内迁到 self.device，结束后立即 offload。
        self.model = self.model.to('cpu')
        self._safeEmptyCudaCache()
        return True

    @staticmethod
    def _safeEmptyCudaCache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _moveModelToDevice(self) -> None:
        self.model = self.model.to(self.device)

    def _offloadModelToCPU(self) -> None:
        self.model = self.model.to('cpu')
        self._safeEmptyCudaCache()

    # ------------------------------------------------------------------
    # Image preprocessing (复刻 vggt_omega.utils.load_fn.load_and_preprocess_images，
    # 但同时返回原图分辨率 RGB 与 metadata，便于后续构建 Camera)
    # ------------------------------------------------------------------
    @staticmethod
    def _loadRGBImage(image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            if image.mode == 'RGBA':
                background = Image.new('RGBA', image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            return image.convert('RGB')

    @staticmethod
    def _cropToSupportedAspectRatio(
        image: Image.Image,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
    ) -> Image.Image:
        width, height = image.size
        aspect_ratio = height / max(width, 1)

        if aspect_ratio < min_aspect_ratio:
            crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
            left = max((width - crop_width) // 2, 0)
            return image.crop((left, 0, left + crop_width, height))

        if aspect_ratio > max_aspect_ratio:
            crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
            top = max((height - crop_height) // 2, 0)
            return image.crop((0, top, width, top + crop_height))

        return image

    @staticmethod
    def _balancedTargetShape(aspect_ratio: float, image_resolution: int, patch_size: int) -> Tuple[int, int]:
        token_number = (image_resolution // patch_size) ** 2
        w_patches = np.sqrt(token_number / aspect_ratio)
        h_patches = token_number / w_patches
        w_patches = max(1, int(np.round(w_patches)))
        h_patches = max(1, int(np.round(h_patches)))
        return h_patches * patch_size, w_patches * patch_size

    @staticmethod
    def _maxSizeTargetShape(aspect_ratio: float, image_resolution: int, patch_size: int) -> Tuple[int, int]:
        if aspect_ratio >= 1.0:
            height = image_resolution
            width = Detector._roundToPatchMultiple(image_resolution / aspect_ratio, patch_size)
        else:
            width = image_resolution
            height = Detector._roundToPatchMultiple(image_resolution * aspect_ratio, patch_size)
        return height, width

    @staticmethod
    def _roundToPatchMultiple(value: float, patch_size: int) -> int:
        return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)

    def _preprocessImages(
        self,
        image_path_list: List[str],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[Dict[str, Any]]]:
        '''按官方 ``load_and_preprocess_images`` 策略对图片进行裁剪/缩放/batch padding，
        并同时返回每张图的 metadata：

        - ``source_image`` (HWC float [0,1]): 仅在极端长宽比时做 center crop 后的原图
          分辨率 RGB（HWC float），用于 ``Camera.loadImage``；
        - ``crop_width`` / ``crop_height``: 裁剪后原图的实际像素分辨率；
        - ``resize_height`` / ``resize_width``: 缩放到 patch 倍数后的尺寸；
        - ``pad_top`` / ``pad_bottom`` / ``pad_left`` / ``pad_right``: batch 内 padding 偏移；
        - ``padded_height`` / ``padded_width``: batch 共同 padded 尺寸。
        '''
        if len(image_path_list) == 0:
            raise ValueError("[Detector::_preprocessImages] At least 1 image is required")

        to_tensor = TF.ToTensor()

        source_images: List[torch.Tensor] = []
        resized_tensors: List[torch.Tensor] = []
        metadata_list: List[Dict[str, Any]] = []
        shapes = set()

        for image_path in image_path_list:
            image = self._cropToSupportedAspectRatio(self._loadRGBImage(image_path))
            crop_w, crop_h = image.size

            # 原图分辨率 RGB（HWC float [0,1]）：用于 Camera.loadImage
            source_chw = to_tensor(image)  # (3, H_crop, W_crop)
            source_hwc = source_chw.permute(1, 2, 0).contiguous()
            source_images.append(source_hwc)

            aspect_ratio = crop_h / max(crop_w, 1)
            if self.preprocess_mode == 'balanced':
                target_h, target_w = self._balancedTargetShape(
                    aspect_ratio, self.image_resolution, self.patch_size,
                )
            else:
                target_h, target_w = self._maxSizeTargetShape(
                    aspect_ratio, self.image_resolution, self.patch_size,
                )

            resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            resized_tensor = to_tensor(resized)
            shapes.add((resized_tensor.shape[1], resized_tensor.shape[2]))
            resized_tensors.append(resized_tensor)

            metadata_list.append({
                'image_path': image_path,
                'crop_width': int(crop_w),
                'crop_height': int(crop_h),
                'resize_height': int(target_h),
                'resize_width': int(target_w),
                'pad_top': 0,
                'pad_bottom': 0,
                'pad_left': 0,
                'pad_right': 0,
                'padded_height': int(target_h),
                'padded_width': int(target_w),
            })

        if len(shapes) > 1:
            warnings.warn(
                f"[Detector] Found images with different shapes: {shapes}; padding to a common size.",
                stacklevel=2,
            )
            max_height = max(s[0] for s in shapes)
            max_width = max(s[1] for s in shapes)
            padded_tensors: List[torch.Tensor] = []
            for tensor, meta in zip(resized_tensors, metadata_list):
                h_padding = max_height - tensor.shape[1]
                w_padding = max_width - tensor.shape[2]
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                if h_padding > 0 or w_padding > 0:
                    tensor = torch.nn.functional.pad(
                        tensor,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode='constant',
                        value=1.0,
                    )
                padded_tensors.append(tensor)
                meta['pad_top'] = int(pad_top)
                meta['pad_bottom'] = int(pad_bottom)
                meta['pad_left'] = int(pad_left)
                meta['pad_right'] = int(pad_right)
                meta['padded_height'] = int(max_height)
                meta['padded_width'] = int(max_width)
            images_tensor = torch.stack(padded_tensors)
        else:
            images_tensor = torch.stack(resized_tensors)

        return source_images, images_tensor, metadata_list

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def _runModel(self, images: torch.Tensor) -> dict:
        with torch.inference_mode():
            return self.model(images)

    def _finalizePredictions(
        self,
        predictions: dict,
        padded_shape_hw: Tuple[int, int],
    ) -> dict:
        '''补上 extrinsic/intrinsic 并把所有 tensor 转成无 batch 维的 numpy。'''
        pose_enc = predictions['pose_enc']
        if not isinstance(pose_enc, torch.Tensor):
            pose_enc = torch.from_numpy(pose_enc)
        pose_enc = pose_enc.to(self.device)

        extrinsic, intrinsic = encoding_to_camera(pose_enc, padded_shape_hw)
        predictions['extrinsic'] = extrinsic
        predictions['intrinsic'] = intrinsic

        for key in list(predictions.keys()):
            value = predictions[key]
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                array = tensor.float().numpy()
                if array.ndim >= 1 and array.shape[0] == 1:
                    array = array[0]
                predictions[key] = array
        return predictions

    @staticmethod
    def _unprojectDepthMapToPointMap(
        depth_map: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        '''与 vggt-omega 官方 demo 一致的 depth -> world points 反投。
        ``extrinsic`` 为 OpenCV camera-from-world (N, 3, 4)，``intrinsic`` 为 (N, 3, 3)，
        ``depth_map`` 为 (N, H, W, 1) 或 (N, H, W)。
        '''
        if depth_map.ndim == 4:
            depth = depth_map[..., 0]
        else:
            depth = depth_map

        num_frames, height, width = depth.shape

        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        x = np.broadcast_to(x[None], (num_frames, height, width))
        y = np.broadcast_to(y[None], (num_frames, height, width))

        fx = intrinsic[:, 0, 0][:, None, None]
        fy = intrinsic[:, 1, 1][:, None, None]
        cx = intrinsic[:, 0, 2][:, None, None]
        cy = intrinsic[:, 1, 2][:, None, None]

        camera_points = np.stack(
            [
                (x - cx) / fx * depth,
                (y - cy) / fy * depth,
                depth,
            ],
            axis=-1,
        )

        rotation = extrinsic[:, :3, :3]
        translation = extrinsic[:, :3, 3]
        return np.einsum(
            'sij,shwj->shwi',
            np.transpose(rotation, (0, 2, 1)),
            camera_points - translation[:, None, None, :],
        )

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
    ) -> Optional[dict]:
        '''输入已经预处理好的 ``(N, 3, H, W)`` tensor，``H/W`` 必须是 patch_size
        的整数倍。返回 predictions dict，关键字段：
          - ``extrinsic`` (N, 3, 4)、``intrinsic`` (N, 3, 3)、``depth`` (N, H, W, 1)、
            ``depth_conf`` (N, H, W)、``images`` (N, 3, H, W)；
          - ``world_points_from_depth`` (N, H, W, 3)；
          - ``valid_indices`` / ``rejected_indices`` 与 input 帧索引一致（保留字段以兼容旧调用方）。
        '''
        if images.ndim != 4 or images.shape[0] == 0:
            print('[ERROR][Detector::detect]')
            print('\t images shape invalid or empty:', tuple(images.shape))
            return None

        H, W = int(images.shape[-2]), int(images.shape[-1])
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            print('[ERROR][Detector::detect]')
            print(
                f'\t image H/W must be divisible by patch_size={self.patch_size}, got {H}x{W}'
            )
            return None

        print(f"Input images shape: {tuple(images.shape)}")

        num_total_images = int(images.shape[0])

        self._moveModelToDevice()
        try:
            images_device = images.to(self.device)
            predictions = self._runModel(images_device)
            predictions = self._finalizePredictions(predictions, (H, W))

            print("Computing world points from depth map...")
            world_points = self._unprojectDepthMapToPointMap(
                predictions['depth'], predictions['extrinsic'], predictions['intrinsic'],
            )
            predictions['world_points_from_depth'] = world_points
            predictions['valid_indices'] = np.arange(num_total_images, dtype=np.int64)
            predictions['rejected_indices'] = np.zeros((0,), dtype=np.int64)
            return predictions
        finally:
            del images_device
            self._offloadModelToCPU()

    # ------------------------------------------------------------------
    # Per-image cropping helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _scaleIntrinsicPaddedToOriginal(
        intrinsic_padded: np.ndarray,
        crop_width: int,
        crop_height: int,
        padded_height: int,
        padded_width: int,
        pad_left: int,
        pad_top: int,
        resize_height: int,
        resize_width: int,
    ) -> np.ndarray:
        '''把在 padded 推理坐标系下的内参 K 重映射到 ``crop_width x crop_height`` 原图坐标系。

        1) 先把 padded K 平移到去掉 batch padding 的 ``resize_height x resize_width`` 坐标系
           （cx -= pad_left, cy -= pad_top）；
        2) 再按 resize -> crop 的等比缩放把 fx, fy, cx, cy 缩到原图分辨率。
        '''
        K = np.asarray(intrinsic_padded, dtype=np.float64).copy()

        K[0, 2] -= float(pad_left)
        K[1, 2] -= float(pad_top)

        sx = float(crop_width) / float(max(resize_width, 1))
        sy = float(crop_height) / float(max(resize_height, 1))

        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        # cx/cy 必须严格在图像中心，Camera.setVGGTPose 假设 width = 2*cx。
        K[0, 2] = float(crop_width) / 2.0
        K[1, 2] = float(crop_height) / 2.0
        return K.astype(np.float32)

    @staticmethod
    def _cropDepthToVisible(
        depth: np.ndarray,
        conf: np.ndarray,
        padded_height: int,
        padded_width: int,
        pad_left: int,
        pad_top: int,
        resize_height: int,
        resize_width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        '''把 padded depth/conf 裁掉 batch padding，只保留 ``resize_height x resize_width`` 可见区域。'''
        H, W = int(depth.shape[0]), int(depth.shape[1])
        scale_y = float(H) / float(max(padded_height, 1))
        scale_x = float(W) / float(max(padded_width, 1))

        y1 = int(round(pad_top * scale_y))
        x1 = int(round(pad_left * scale_x))
        y2 = int(round((pad_top + resize_height) * scale_y))
        x2 = int(round((pad_left + resize_width) * scale_x))

        y1 = max(0, min(y1, H))
        x1 = max(0, min(x1, W))
        y2 = max(0, min(y2, H))
        x2 = max(0, min(x2, W))

        if x2 <= x1 or y2 <= y1:
            print('[WARN][Detector::_cropDepthToVisible]')
            print(
                f'\t empty crop region (x1={x1}, y1={y1}, x2={x2}, y2={y2}); '
                f'falling back to full {H}x{W} depth.'
            )
            return depth, conf

        return depth[y1:y2, x1:x2], conf[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Public entries: images / image files / folder / video
    # ------------------------------------------------------------------
    @torch.no_grad()
    def detectImages(
        self,
        images: torch.Tensor,
        source_images: List[torch.Tensor],
        metadata_list: List[Dict[str, Any]],
    ) -> Optional[List[Camera]]:
        '''Args:
            images: ``(N, 3, H, W)`` 已经按 VGGT-Omega 预处理（含 resize + batch padding）的 tensor，
                ``H/W`` 为 ``patch_size`` 的整数倍。
            source_images: 长度 N 的列表，元素为 ``(H_i, W_i, 3)`` float [0, 1] 原图 RGB，
                即去掉极端长宽比 center crop 后的可见区域。
            metadata_list: 长度 N，由 ``_preprocessImages`` 生成，记录 crop/resize/padding 偏移。
        '''
        if images.shape[0] == 0:
            print('[WARN][Detector::detectImages]')
            print('\t images are empty!')
            return None

        if len(source_images) != images.shape[0] or len(metadata_list) != images.shape[0]:
            print('[ERROR][Detector::detectImages]')
            print(
                '\t len mismatch: source_images=', len(source_images),
                'metadata=', len(metadata_list), 'images=', images.shape[0],
            )
            return None

        predictions = self.detect(images)
        if predictions is None:
            return None

        extrinsics = predictions['extrinsic']     # (N, 3, 4) padded
        intrinsics_pad = predictions['intrinsic']  # (N, 3, 3) padded
        depths = predictions['depth']              # (N, H, W, 1) or (N, H, W)
        depth_conf = predictions['depth_conf']     # (N, H, W)

        if depths.ndim == 4:
            depths_2d = depths.reshape(depths.shape[0], depths.shape[1], depths.shape[2])
        else:
            depths_2d = depths

        extr_dtype = extrinsics.dtype if hasattr(extrinsics, 'dtype') else np.float32

        print('start create cameras...')
        camera_list: List[Camera] = []
        original_intrinsic_list: List[np.ndarray] = []
        for i in range(extrinsics.shape[0]):
            extrinsic_44 = np.zeros((4, 4), dtype=extr_dtype)
            extrinsic_44[:3, :4] = extrinsics[i]
            extrinsic_44[3, 3] = 1.0

            meta = metadata_list[i]
            crop_w = int(meta['crop_width'])
            crop_h = int(meta['crop_height'])

            original_intrinsic = self._scaleIntrinsicPaddedToOriginal(
                intrinsics_pad[i],
                crop_width=crop_w,
                crop_height=crop_h,
                padded_height=int(meta['padded_height']),
                padded_width=int(meta['padded_width']),
                pad_left=int(meta['pad_left']),
                pad_top=int(meta['pad_top']),
                resize_height=int(meta['resize_height']),
                resize_width=int(meta['resize_width']),
            )
            original_intrinsic_list.append(original_intrinsic)

            camera = Camera.fromVGGTPose(extrinsic_44, original_intrinsic, device='cpu')
            # 默认 6 位序号 + .png 作为 image_id；事后由 detectImageFiles 用真实文件名覆盖。
            camera.image_id = f'{(i+1):06d}.png'
            camera.loadImage(source_images[i])

            cropped_depth, cropped_conf = self._cropDepthToVisible(
                depths_2d[i], depth_conf[i],
                padded_height=int(meta['padded_height']),
                padded_width=int(meta['padded_width']),
                pad_left=int(meta['pad_left']),
                pad_top=int(meta['pad_top']),
                resize_height=int(meta['resize_height']),
                resize_width=int(meta['resize_width']),
            )
            camera.loadDepth(cropped_depth, cropped_conf)
            camera_list.append(camera)

        # 用 camera_list 反投得到的稀疏 3D 点云供可视化/COLMAP 输出使用。
        pcd = CameraConvertor.createDepthPcd(camera_list)
        predictions['points'] = np.asarray(pcd.points, dtype=np.float32)
        predictions['colors'] = (
            np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        )

        if len(original_intrinsic_list) > 0:
            predictions['original_intrinsic'] = np.stack(
                original_intrinsic_list, axis=0,
            ).astype(np.float32, copy=False)

        return camera_list

    @torch.no_grad()
    def detectImageFiles(
        self,
        image_file_path_list: list,
    ) -> Optional[List[Camera]]:
        '''按 VGGT-Omega 官方策略对图片做 resize + batch padding 后调用 ``detectImages``。'''
        if len(image_file_path_list) == 0:
            print('[WARN][Detector::detectImageFiles]')
            print('\t images are empty!')
            return None

        print(f"Found {len(image_file_path_list)} images")

        source_images, images_tensor, metadata_list = self._preprocessImages(
            image_file_path_list,
        )

        if images_tensor.shape[0] == 0:
            print('[WARN][Detector::detectImageFiles]')
            print('\t images not found!')
            return None

        camera_list = self.detectImages(images_tensor, source_images, metadata_list)
        if camera_list is None:
            return None

        # 用真实文件名（仅 basename）覆盖 detectImages 内部默认的 ``f'{i:06d}.png'``。
        for camera, image_file_path in zip(camera_list, image_file_path_list):
            camera.image_id = os.path.basename(image_file_path)

        return camera_list

    @torch.no_grad()
    def detectImageFolder(
        self,
        image_folder_path: str,
    ) -> Optional[List[Camera]]:
        if not os.path.exists(image_folder_path):
            print('[ERROR][Detector::detectImageFolder]')
            print('\t image folder not exist!')
            print('\t image_folder_path:', image_folder_path)
            return None

        image_file_name_list = sorted(os.listdir(image_folder_path))
        image_file_path_list = [
            os.path.join(image_folder_path, name)
            for name in image_file_name_list
            if name.split('.')[-1].lower() in ('png', 'jpg', 'jpeg')
        ]

        return self.detectImageFiles(image_file_path_list)

    @torch.no_grad()
    def detectVideoFile(
        self,
        video_file_path: str,
        save_image_folder_path: str,
        target_image_num: int = 200,
    ) -> Optional[List[Camera]]:
        if not os.path.exists(video_file_path):
            print('[ERROR][Detector::detectVideoFile]')
            print('\t video file not exist!')
            print('\t video_file_path:', video_file_path)
            return None

        if os.path.exists(save_image_folder_path):
            rmtree(save_image_folder_path)
        os.makedirs(save_image_folder_path, exist_ok=True)

        if not videoToImages(
            video_file_path,
            save_image_folder_path,
            target_image_num=target_image_num,
            scale=1,
            print_progress=True,
        ):
            print('[ERROR][Detector::detectVideoFile]')
            print('\t videoToImages failed!')
            print('\t video_file_path:', video_file_path)
            return None

        return self.detectImageFolder(save_image_folder_path)
