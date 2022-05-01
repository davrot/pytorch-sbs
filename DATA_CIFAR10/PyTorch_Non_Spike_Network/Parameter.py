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

# %%
from dataclasses import dataclass, field
import numpy as np
import torch
import os


@dataclass
class Network:
    """Parameters of the network. The details about
    its layers and the number of output neurons."""

    number_of_output_neurons: int = field(default=0)
    forward_kernel_size: list[list[int]] = field(default_factory=list)
    forward_neuron_numbers: list[list[int]] = field(default_factory=list)
    strides: list[list[int]] = field(default_factory=list)
    dilation: list[list[int]] = field(default_factory=list)
    padding: list[list[int]] = field(default_factory=list)
    is_pooling_layer: list[bool] = field(default_factory=list)
    w_trainable: list[bool] = field(default_factory=list)
    eps_xy_trainable: list[bool] = field(default_factory=list)
    eps_xy_mean: list[bool] = field(default_factory=list)


@dataclass
class LearningParameters:
    """Parameter required for training"""

    loss_coeffs_mse: float = field(default=0.5)
    loss_coeffs_kldiv: float = field(default=1.0)
    learning_rate_gamma_w: float = field(default=-1.0)
    learning_rate_gamma_eps_xy: float = field(default=-1.0)
    learning_rate_threshold_w: float = field(default=0.00001)
    learning_rate_threshold_eps_xy: float = field(default=0.00001)
    learning_active: bool = field(default=True)
    weight_noise_amplitude: float = field(default=0.01)
    eps_xy_intitial: float = field(default=0.1)
    test_every_x_learning_steps: int = field(default=50)
    test_during_learning: bool = field(default=True)
    lr_scheduler_factor: float = field(default=0.75)
    lr_scheduler_patience: int = field(default=10)
    optimizer_name: str = field(default="Adam")
    lr_schedule_name: str = field(default="ReduceLROnPlateau")
    number_of_batches_for_one_update: int = field(default=1)
    alpha_number_of_iterations: int = field(default=0)
    overload_path: str = field(default="./Previous")


@dataclass
class Augmentation:
    """Parameters used for data augmentation."""

    crop_width_in_pixel: int = field(default=2)
    flip_p: float = field(default=0.5)
    jitter_brightness: float = field(default=0.5)
    jitter_contrast: float = field(default=0.1)
    jitter_saturation: float = field(default=0.1)
    jitter_hue: float = field(default=0.15)


@dataclass
class ImageStatistics:
    """(Statistical) information about the input. i.e.
    mean values and the x and y size of the input"""

    mean: list[float] = field(default_factory=list)
    the_size: list[int] = field(default_factory=list)


@dataclass
class Config:
    """Master config class."""

    # Sub classes
    network_structure: Network = field(default_factory=Network)
    learning_parameters: LearningParameters = field(default_factory=LearningParameters)
    augmentation: Augmentation = field(default_factory=Augmentation)
    image_statistics: ImageStatistics = field(default_factory=ImageStatistics)

    batch_size: int = field(default=500)
    data_mode: str = field(default="")

    learning_step: int = field(default=0)
    learning_step_max: int = field(default=10000)

    number_of_cpu_processes: int = field(default=-1)

    number_of_spikes: int = field(default=0)
    cooldown_after_number_of_spikes: int = field(default=0)

    weight_path: str = field(default="./Weights/")
    eps_xy_path: str = field(default="./EpsXY/")
    data_path: str = field(default="./")

    reduction_cooldown: float = field(default=25.0)
    epsilon_0: float = field(default=1.0)

    update_after_x_batch: float = field(default=1.0)

    def __post_init__(self) -> None:
        """Post init determines the number of cores.
        Creates the required directory and gives us an optimized
        (for the amount of cores) batch size."""
        number_of_cpu_processes_temp = os.cpu_count()

        if self.number_of_cpu_processes < 1:
            if number_of_cpu_processes_temp is None:
                self.number_of_cpu_processes = 1
            else:
                self.number_of_cpu_processes = number_of_cpu_processes_temp

        os.makedirs(self.weight_path, exist_ok=True)
        os.makedirs(self.eps_xy_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)

        self.batch_size = (
            self.batch_size // self.number_of_cpu_processes
        ) * self.number_of_cpu_processes

        self.batch_size = np.max((self.batch_size, self.number_of_cpu_processes))
        self.batch_size = int(self.batch_size)

    def get_epsilon_t(self):
        """Generates the time series of the basic epsilon."""
        np_epsilon_t: np.ndarray = np.ones((self.number_of_spikes), dtype=np.float32)
        np_epsilon_t[
            self.cooldown_after_number_of_spikes : self.number_of_spikes
        ] /= self.reduction_cooldown
        return torch.tensor(np_epsilon_t)

    def get_update_after_x_pattern(self):
        """Tells us after how many pattern we need to update the weights."""
        return self.batch_size * self.update_after_x_batch
