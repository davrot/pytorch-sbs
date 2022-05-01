# MIT License
# Copyright 2022 University of Bremen
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#
# David Rotermund ( davrot@uni-bremen.de )
#
#
# Release history:
# ================
# 1.0.0 -- 01.05.2022: first release
#
#

from abc import ABC, abstractmethod
import torch
import numpy as np
import torchvision as tv  # type: ignore
from Parameter import Config


class DatasetMaster(torch.utils.data.Dataset, ABC):

    path_label: str
    label_storage: np.ndarray
    pattern_storage: np.ndarray
    number_of_pattern: int
    mean: list[float]

    # Initialize
    def __init__(
        self,
        train: bool = False,
        path_pattern: str = "./",
        path_label: str = "./",
    ) -> None:
        super().__init__()

        if train is True:
            self.label_storage = np.load(path_label + "/TrainLabelStorage.npy")
        else:
            self.label_storage = np.load(path_label + "/TestLabelStorage.npy")

        if train is True:
            self.pattern_storage = np.load(path_pattern + "/TrainPatternStorage.npy")
        else:
            self.pattern_storage = np.load(path_pattern + "/TestPatternStorage.npy")

        self.number_of_pattern = self.label_storage.shape[0]

        self.mean = []

    def __len__(self) -> int:
        return self.number_of_pattern

    # Get one pattern at position index
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pass

    @abstractmethod
    def pattern_filter_test(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        pass

    @abstractmethod
    def pattern_filter_train(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        pass


class DatasetMNIST(DatasetMaster):
    """Contstructor"""

    # Initialize
    def __init__(
        self,
        train: bool = False,
        path_pattern: str = "./",
        path_label: str = "./",
    ) -> None:
        super().__init__(train, path_pattern, path_label)

        self.pattern_storage = np.ascontiguousarray(
            self.pattern_storage[:, np.newaxis, :, :].astype(dtype=np.float32)
        )

        self.pattern_storage /= np.max(self.pattern_storage)

        mean = self.pattern_storage.mean(3).mean(2).mean(0)
        self.mean = [*mean]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:

        image = self.pattern_storage[index, 0:1, :, :]
        target = int(self.label_storage[index])
        return torch.tensor(image), target

    def pattern_filter_test(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        """0. The test image comes in
        1. is center cropped
        2. on/off filteres
        3. returned.

        This is a 1 channel version (e.g. one gray channel).
        """

        assert len(cfg.image_statistics.mean) == 1
        assert len(cfg.image_statistics.the_size) == 2
        assert cfg.image_statistics.the_size[0] > 0
        assert cfg.image_statistics.the_size[1] > 0

        # Transformation chain
        my_transforms: torch.nn.Sequential = torch.nn.Sequential(
            tv.transforms.CenterCrop(size=cfg.image_statistics.the_size),
        )
        scripted_transforms = torch.jit.script(my_transforms)

        # Preprocess the input data
        pattern = scripted_transforms(pattern)

        # => On/Off
        if cfg.augmentation.use_on_off_filter is True:
            my_on_off_filter: OnOffFilter = OnOffFilter(p=cfg.image_statistics.mean[0])
            gray: torch.Tensor = my_on_off_filter(
                pattern[:, 0:1, :, :],
            )
        else:
            gray = pattern[:, 0:1, :, :] + torch.finfo(torch.float32).eps

        return gray

    def pattern_filter_train(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        """0. The training image comes in
        1. is cropped from a random position
        2. on/off filteres
        3. returned.

        This is a 1 channel version (e.g. one gray channel).
        """

        assert len(cfg.image_statistics.mean) == 1
        assert len(cfg.image_statistics.the_size) == 2
        assert cfg.image_statistics.the_size[0] > 0
        assert cfg.image_statistics.the_size[1] > 0

        # Transformation chain
        my_transforms: torch.nn.Sequential = torch.nn.Sequential(
            tv.transforms.RandomCrop(size=cfg.image_statistics.the_size),
        )
        scripted_transforms = torch.jit.script(my_transforms)

        # Preprocess the input data
        pattern = scripted_transforms(pattern)

        # => On/Off
        if cfg.augmentation.use_on_off_filter is True:
            my_on_off_filter: OnOffFilter = OnOffFilter(p=cfg.image_statistics.mean[0])
            gray: torch.Tensor = my_on_off_filter(
                pattern[:, 0:1, :, :],
            )
        else:
            gray = pattern[:, 0:1, :, :] + torch.finfo(torch.float32).eps

        return gray


class DatasetFashionMNIST(DatasetMaster):
    """Contstructor"""

    # Initialize
    def __init__(
        self,
        train: bool = False,
        path_pattern: str = "./",
        path_label: str = "./",
    ) -> None:
        super().__init__(train, path_pattern, path_label)

        self.pattern_storage = np.ascontiguousarray(
            self.pattern_storage[:, np.newaxis, :, :].astype(dtype=np.float32)
        )

        self.pattern_storage /= np.max(self.pattern_storage)

        mean = self.pattern_storage.mean(3).mean(2).mean(0)
        self.mean = [*mean]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:

        image = self.pattern_storage[index, 0:1, :, :]
        target = int(self.label_storage[index])
        return torch.tensor(image), target

    def pattern_filter_test(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        """0. The test image comes in
        1. is center cropped
        2. on/off filteres
        3. returned.

        This is a 1 channel version (e.g. one gray channel).
        """

        assert len(cfg.image_statistics.mean) == 1
        assert len(cfg.image_statistics.the_size) == 2
        assert cfg.image_statistics.the_size[0] > 0
        assert cfg.image_statistics.the_size[1] > 0

        # Transformation chain
        my_transforms: torch.nn.Sequential = torch.nn.Sequential(
            tv.transforms.CenterCrop(size=cfg.image_statistics.the_size),
        )
        scripted_transforms = torch.jit.script(my_transforms)

        # Preprocess the input data
        pattern = scripted_transforms(pattern)

        # => On/Off
        if cfg.augmentation.use_on_off_filter is True:
            my_on_off_filter: OnOffFilter = OnOffFilter(p=cfg.image_statistics.mean[0])
            gray: torch.Tensor = my_on_off_filter(
                pattern[:, 0:1, :, :],
            )
        else:
            gray = pattern[:, 0:1, :, :] + torch.finfo(torch.float32).eps

        return gray

    def pattern_filter_train(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        """0. The training image comes in
        1. is cropped from a random position
        2. on/off filteres
        3. returned.

        This is a 1 channel version (e.g. one gray channel).
        """

        assert len(cfg.image_statistics.mean) == 1
        assert len(cfg.image_statistics.the_size) == 2
        assert cfg.image_statistics.the_size[0] > 0
        assert cfg.image_statistics.the_size[1] > 0

        # Transformation chain
        my_transforms: torch.nn.Sequential = torch.nn.Sequential(
            tv.transforms.RandomCrop(size=cfg.image_statistics.the_size),
            tv.transforms.RandomHorizontalFlip(p=cfg.augmentation.flip_p),
            tv.transforms.ColorJitter(
                brightness=cfg.augmentation.jitter_brightness,
                contrast=cfg.augmentation.jitter_contrast,
                saturation=cfg.augmentation.jitter_saturation,
                hue=cfg.augmentation.jitter_hue,
            ),
        )
        scripted_transforms = torch.jit.script(my_transforms)

        # Preprocess the input data
        pattern = scripted_transforms(pattern)

        # => On/Off
        if cfg.augmentation.use_on_off_filter is True:
            my_on_off_filter: OnOffFilter = OnOffFilter(p=cfg.image_statistics.mean[0])
            gray: torch.Tensor = my_on_off_filter(
                pattern[:, 0:1, :, :],
            )
        else:
            gray = pattern[:, 0:1, :, :] + torch.finfo(torch.float32).eps

        return gray


class DatasetCIFAR(DatasetMaster):
    """Contstructor"""

    # Initialize
    def __init__(
        self,
        train: bool = False,
        path_pattern: str = "./",
        path_label: str = "./",
    ) -> None:
        super().__init__(train, path_pattern, path_label)

        self.pattern_storage = np.ascontiguousarray(
            np.moveaxis(self.pattern_storage.astype(dtype=np.float32), 3, 1)
        )
        self.pattern_storage /= np.max(self.pattern_storage)

        mean = self.pattern_storage.mean(3).mean(2).mean(0)
        self.mean = [*mean]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:

        image = self.pattern_storage[index, :, :, :]
        target = int(self.label_storage[index])
        return torch.tensor(image), target

    def pattern_filter_test(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        """0. The test image comes in
        1. is center cropped
        2. on/off filteres
        3. returned.

        This is a 3 channel version (e.g. r,g,b channels).
        """

        assert len(cfg.image_statistics.mean) == 3
        assert len(cfg.image_statistics.the_size) == 2
        assert cfg.image_statistics.the_size[0] > 0
        assert cfg.image_statistics.the_size[1] > 0

        # Transformation chain
        my_transforms: torch.nn.Sequential = torch.nn.Sequential(
            tv.transforms.CenterCrop(size=cfg.image_statistics.the_size),
        )
        scripted_transforms = torch.jit.script(my_transforms)

        # Preprocess the input data
        pattern = scripted_transforms(pattern)

        # => On/Off
        if cfg.augmentation.use_on_off_filter is True:
            my_on_off_filter_r: OnOffFilter = OnOffFilter(
                p=cfg.image_statistics.mean[0]
            )
            my_on_off_filter_g: OnOffFilter = OnOffFilter(
                p=cfg.image_statistics.mean[1]
            )
            my_on_off_filter_b: OnOffFilter = OnOffFilter(
                p=cfg.image_statistics.mean[2]
            )
            r: torch.Tensor = my_on_off_filter_r(
                pattern[:, 0:1, :, :],
            )
            g: torch.Tensor = my_on_off_filter_g(
                pattern[:, 1:2, :, :],
            )
            b: torch.Tensor = my_on_off_filter_b(
                pattern[:, 2:3, :, :],
            )
        else:
            r = pattern[:, 0:1, :, :] + torch.finfo(torch.float32).eps
            g = pattern[:, 1:2, :, :] + torch.finfo(torch.float32).eps
            b = pattern[:, 2:3, :, :] + torch.finfo(torch.float32).eps

        new_tensor: torch.Tensor = torch.cat((r, g, b), dim=1)
        return new_tensor

    def pattern_filter_train(self, pattern: torch.Tensor, cfg: Config) -> torch.Tensor:
        """0. The training image comes in
        1. is cropped from a random position
        2. is randomly horizontally flipped
        3. is randomly color jitteres
        4. on/off filteres
        5. returned.

        This is a 3 channel version (e.g. r,g,b channels).
        """
        assert len(cfg.image_statistics.mean) == 3
        assert len(cfg.image_statistics.the_size) == 2
        assert cfg.image_statistics.the_size[0] > 0
        assert cfg.image_statistics.the_size[1] > 0

        # Transformation chain
        my_transforms: torch.nn.Sequential = torch.nn.Sequential(
            tv.transforms.RandomCrop(size=cfg.image_statistics.the_size),
            tv.transforms.RandomHorizontalFlip(p=cfg.augmentation.flip_p),
            tv.transforms.ColorJitter(
                brightness=cfg.augmentation.jitter_brightness,
                contrast=cfg.augmentation.jitter_contrast,
                saturation=cfg.augmentation.jitter_saturation,
                hue=cfg.augmentation.jitter_hue,
            ),
        )
        scripted_transforms = torch.jit.script(my_transforms)

        # Preprocess the input data
        pattern = scripted_transforms(pattern)

        # => On/Off
        if cfg.augmentation.use_on_off_filter is True:
            my_on_off_filter_r: OnOffFilter = OnOffFilter(
                p=cfg.image_statistics.mean[0]
            )
            my_on_off_filter_g: OnOffFilter = OnOffFilter(
                p=cfg.image_statistics.mean[1]
            )
            my_on_off_filter_b: OnOffFilter = OnOffFilter(
                p=cfg.image_statistics.mean[2]
            )
            r: torch.Tensor = my_on_off_filter_r(
                pattern[:, 0:1, :, :],
            )
            g: torch.Tensor = my_on_off_filter_g(
                pattern[:, 1:2, :, :],
            )
            b: torch.Tensor = my_on_off_filter_b(
                pattern[:, 2:3, :, :],
            )
        else:
            r = pattern[:, 0:1, :, :] + torch.finfo(torch.float32).eps
            g = pattern[:, 1:2, :, :] + torch.finfo(torch.float32).eps
            b = pattern[:, 2:3, :, :] + torch.finfo(torch.float32).eps

        new_tensor: torch.Tensor = torch.cat((r, g, b), dim=1)
        return new_tensor


class OnOffFilter(torch.nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super(OnOffFilter, self).__init__()
        self.p: float = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        assert tensor.shape[1] == 1

        tensor_clone = 2.0 * (tensor - self.p)

        temp_0: torch.Tensor = torch.where(
            tensor_clone < 0.0,
            -tensor_clone,
            tensor_clone.new_zeros(tensor_clone.shape, dtype=tensor_clone.dtype),
        )

        temp_1: torch.Tensor = torch.where(
            tensor_clone >= 0.0,
            tensor_clone,
            tensor_clone.new_zeros(tensor_clone.shape, dtype=tensor_clone.dtype),
        )

        new_tensor: torch.Tensor = torch.cat((temp_0, temp_1), dim=1)

        return new_tensor

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(p={0})".format(self.p)


if __name__ == "__main__":
    pass
