import random
import numpy as np
from typing import List, Dict, Hashable, Mapping

from monai.data import get_track_meta
from monai.config import KeysCollection, DtypeLike, NdarrayOrTensor
from monai.utils import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_tensor
from monai.transforms import MapTransform, GaussianSmoothd, Resized, RandAdjustContrastd
from monai.transforms.transform import Transform
from monai.transforms.utils_pytorch_numpy_unification import clip

import torch
import torch.distributed as dist
import torch.nn.functional as F


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def Dice(image: torch.Tensor, mask: torch.Tensor, eps=1e-5, Binary=False, Sigmoid=False):
    """
        calculate the dice, active effective pixel
    """
    reduce_axis: List[int] = torch.arange(2, len(image.shape)).tolist()
    with torch.no_grad():
        if Sigmoid:
            image = torch.sigmoid(image)

        if Binary:
            image = (image >= 0.5) * 1.0

        inter = torch.sum(image * mask, dim=reduce_axis)
        union = torch.sum(image, dim=reduce_axis) + torch.sum(mask, dim=reduce_axis)

        dice = (2 * inter + eps) / (union + eps)

        return torch.mean(dice)


def distributed_all_gather(tensor_list,
                           valid_batch_size=None,
                           out_numpy=False,
                           world_size=None,
                           no_barrier=False,
                           is_valid=None):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


class VolumeFeatureMixed(object):
    """
        Extract the volume's different feature and mix it up
    """
    def __init__(self, visualize=False):
        self.visualize = visualize

    @staticmethod
    def MedianFilter(x, kernel_size=7):
        # Using the Pytorch's Average Pooling Function to achieve
        if len(x.shape) != 5:
            while len(x.shape) < 5:
                x = x.unsqueeze(0)

        padding = kernel_size // 2

        x = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
        x = x.squeeze()

        return x

    @staticmethod
    def SobelFilter(v_):
        if len(v_.shape) != 4:
            while len(v_.shape) < 4:
                v_ = v_.unsqueeze(0)

        d, h, w = v_.shape[1:]
        # default the Sobel kernel
        Sobel_v = torch.tensor([[-1, -2, 0, 2, 1],
                                [-4, -8, 0, 8, 4],
                                [-6, -12, 0, 12, 6],
                                [-4, -8, 0, 8, 4],
                                [-1, -2, 0, 2, 1]], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)

        Sobel_h = torch.tensor([[-1, -4, -6, -4, -1],
                                [-2, -8, -12, -8, -2],
                                [0, 0, 0, 0, 0],
                                [2, 8, 12, 8, 2],
                                [1, 4, 6, 4, 1]], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)

        x = torch.sqrt(
            F.conv2d(v_, Sobel_v.repeat(d, 1, 1, 1), stride=1, padding=2, groups=d) ** 2 +
            F.conv2d(v_, Sobel_h.repeat(d, 1, 1, 1), stride=1, padding=2, groups=d) ** 2
        )

        y = torch.sqrt(
            F.conv2d(v_.transpose(1, 2), Sobel_v.repeat(h, 1, 1, 1), stride=1, padding=2, groups=h) ** 2 +
            F.conv2d(v_.transpose(1, 2), Sobel_h.repeat(h, 1, 1, 1), stride=1, padding=2, groups=h) ** 2
        ).transpose(2, 1)

        z = torch.sqrt(
            F.conv2d(v_.transpose(1, 3), Sobel_v.repeat(w, 1, 1, 1), stride=1, padding=2, groups=w) ** 2 +
            F.conv2d(v_.transpose(1, 3), Sobel_h.repeat(w, 1, 1, 1), stride=1, padding=2, groups=w) ** 2
        ).transpose(1, 3)

        v_ = ((x + y + z) / 3).squeeze()

        return v_

    @staticmethod
    def NMS(step=np.pi/4):
        k0 = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        k1 = [[0, 0, -1], [0, 1, 0], [0, 0, 0]]
        k2 = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
        k3 = [[-1, 0, 0], [0, 1, 0], [0, 0, 0]]
        k4 = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        k5 = [[0, 0, 0], [0, 1, 0], [-1, 0, 0]]
        k6 = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        k7 = [[0, 0, 0], [0, 1, 0], [0, 0, -1]]

        kernel_size = 3

        return

    @staticmethod
    def MorphologyErosion(x, kernel_size=3):
        # Using the Pytorch's Max Pooling Function to achieve
        if len(x.shape) != 5:
            while len(x.shape) < 5:
                x = x.unsqueeze(0)

        padding = kernel_size // 2

        x = F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
        x = x.squeeze()
        return x

    def __call__(self, x, *args, **kwargs):
        x = torch.from_numpy(x)

        M_ = self.MedianFilter(x)
        S_ = self.SobelFilter(M_)
        out = 255 * (S_ - torch.min(S_)) / (torch.max(S_) - torch.min(S_)) + \
              255 * (M_ - torch.min(M_)) / (torch.max(M_) - torch.min(M_))

        return out.detach().numpy()


class GaussianSmoothd_:
    def __init__(self, keys="image", sigma=random.uniform(0.5, 1.), approx='erf', prob=0.2):
        self.GaussianSmooth = GaussianSmoothd(keys=keys, sigma=sigma, approx=approx)
        self.prob = prob

    def __call__(self, data):
        if np.random.uniform() <= self.prob:
            if isinstance(data, list):
                data = data[0]

            data = self.GaussianSmooth(data)
        return data


class Resized_:
    def __init__(self, keys="image", mode1="trilinear", mode2="nearest",
                 spatial_size=(119, 112, 112), zoom_range=(0.5, 1), prob=0.25):
        target_size = np.round(np.array(spatial_size) * random.uniform(*zoom_range)).astype(int)

        self.Resize_1 = Resized(keys=keys, spatial_size=target_size, mode=mode1)
        self.Resize_2 = Resized(keys=keys, spatial_size=spatial_size, mode=mode2)
        self.prob = prob

    def __call__(self, data):
        if np.random.uniform() <= self.prob:
            if isinstance(data, list):
                data = data[0]

            data = self.Resize_1(data)
            data = self.Resize_2(data)
        return data


class GammaTransform:
    def __init__(self, keys="image", gamma=(0.7, 1.5), invert=False, prob=0.3):
        self.GammaTransform = RandAdjustContrastd(keys=keys, prob=0.3, gamma=gamma)
        self.invert = invert
        self.prob = prob

    def __call__(self, data):
        if np.random.uniform() <= self.prob:
            if isinstance(data, list):
                data = data[0]

            if self.invert:
                data["image"] = -data["image"]
                data = self.GammaTransform(data)
                data["image"] = -data["image"]
            else:
                data = self.GammaTransform(data)
        return data


class ScaleIntensityRange(Transform):
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    When `b_min` or `b_max` are `None`, `scaled_array * (b_max - b_min) + b_min` will be skipped.
    If `clip=True`, when `b_min`/`b_max` is None, the clipping is not performed on the corresponding edge.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        clip_: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        a_min: float = 0.75,
        a_max: float = 1.25,
        clip_: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.clip = clip_
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype

        min_ = img.min()
        max_ = img.max()
        mn = img.mean()
        factor = random.uniform(self.a_min, self.a_max)

        img = (img - mn)
        img = img * factor + mn

        if self.clip:
            img = clip(img, min_, max_)
        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]

        return ret


class ScaleIntensityRanged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        clip_: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        clip_: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, clip_, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class ScaleIntensityRanged_:
    def __init__(self, keys="image", a_min=0.75, a_max=1.25, clip_=True, prob=0.15):
        self.ScaleIntensityRange = ScaleIntensityRanged(keys=keys, a_min=a_min, a_max=a_max, clip_=clip_)
        self.prob = prob

    def __call__(self, data):
        if np.random.uniform() <= self.prob:
            if isinstance(data, list):
                data = data[0]

            data = self.ScaleIntensityRange(data)
        return data
