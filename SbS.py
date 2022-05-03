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
import torch
import numpy as np

try:
    import PySpikeGeneration2DManyIP

    cpp_spike: bool = True
except Exception:
    cpp_spike = False

try:
    import PyHDynamicCNNManyIP

    cpp_sbs: bool = True
except Exception:
    cpp_sbs = False


class SbS(torch.nn.Module):

    _epsilon_xy: torch.nn.parameter.Parameter
    _epsilon_xy_exists: bool = False
    _epsilon_0: torch.Tensor | None = None
    _epsilon_t: torch.Tensor | None = None
    _weights: torch.nn.parameter.Parameter
    _weights_exists: bool = False
    _kernel_size: torch.Tensor | None = None
    _stride: torch.Tensor | None = None
    _dilation: torch.Tensor | None = None
    _padding: torch.Tensor | None = None
    _output_size: torch.Tensor | None = None
    _number_of_spikes: torch.Tensor | None = None
    _number_of_cpu_processes: torch.Tensor | None = None
    _number_of_neurons: torch.Tensor | None = None
    _number_of_input_neurons: torch.Tensor | None = None
    _h_initial: torch.Tensor | None = None
    _epsilon_xy_backup: torch.Tensor | None = None
    _weights_backup: torch.Tensor | None = None
    _alpha_number_of_iterations: torch.Tensor | None = None

    def __init__(
        self,
        number_of_input_neurons: int,
        number_of_neurons: int,
        input_size: list[int],
        forward_kernel_size: list[int],
        number_of_spikes: int,
        epsilon_t: torch.Tensor,
        epsilon_xy_intitial: float = 0.1,
        epsilon_0: float = 1.0,
        weight_noise_amplitude: float = 0.01,
        is_pooling_layer: bool = False,
        strides: list[int] = [1, 1],
        dilation: list[int] = [0, 0],
        padding: list[int] = [0, 0],
        alpha_number_of_iterations: int = 0,
        number_of_cpu_processes: int = 1,
    ) -> None:
        """Constructor"""
        super().__init__()

        self.stride = torch.tensor(strides, dtype=torch.int64)

        self.dilation = torch.tensor(dilation, dtype=torch.int64)

        self.padding = torch.tensor(padding, dtype=torch.int64)

        self.kernel_size = torch.tensor(
            forward_kernel_size,
            dtype=torch.int64,
        )

        self.number_of_input_neurons = torch.tensor(
            number_of_input_neurons,
            dtype=torch.int64,
        )

        self.number_of_neurons = torch.tensor(
            number_of_neurons,
            dtype=torch.int64,
        )

        self.alpha_number_of_iterations = torch.tensor(
            alpha_number_of_iterations, dtype=torch.int64
        )

        self.calculate_output_size(torch.tensor(input_size, dtype=torch.int64))

        self.set_h_init_to_uniform()

        self.initialize_epsilon_xy(epsilon_xy_intitial)

        self.epsilon_0 = torch.tensor(epsilon_0, dtype=torch.float64)

        self.number_of_cpu_processes = torch.tensor(
            number_of_cpu_processes, dtype=torch.int64
        )

        self.number_of_spikes = torch.tensor(number_of_spikes, dtype=torch.int64)

        self.epsilon_t = epsilon_t.type(dtype=torch.float64)

        self.initialize_weights(
            is_pooling_layer=is_pooling_layer,
            noise_amplitude=weight_noise_amplitude,
        )

        self.functional_sbs = FunctionalSbS.apply

    ####################################################################
    # Variables in and out                                             #
    ####################################################################

    @property
    def epsilon_xy(self) -> torch.Tensor | None:
        if self._epsilon_xy_exists is False:
            return None
        else:
            return self._epsilon_xy

    @epsilon_xy.setter
    def epsilon_xy(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 4
        assert value.dtype == torch.float64
        if self._epsilon_xy_exists is False:
            self._epsilon_xy = torch.nn.parameter.Parameter(
                value.detach().clone(memory_format=torch.contiguous_format),
                requires_grad=True,
            )
            self._epsilon_xy_exists = True
        else:
            self._epsilon_xy.data = value.detach().clone(
                memory_format=torch.contiguous_format
            )

    @property
    def epsilon_0(self) -> torch.Tensor | None:
        return self._epsilon_0

    @epsilon_0.setter
    def epsilon_0(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert torch.numel(value) == 1
        assert value.dtype == torch.float64
        assert value.item() > 0
        self._epsilon_0 = value.detach().clone(memory_format=torch.contiguous_format)
        self._epsilon_0.requires_grad_(False)

    @property
    def epsilon_t(self) -> torch.Tensor | None:
        return self._epsilon_t

    @epsilon_t.setter
    def epsilon_t(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert value.dtype == torch.float64
        self._epsilon_t = value.detach().clone(memory_format=torch.contiguous_format)
        self._epsilon_t.requires_grad_(False)

    @property
    def weights(self) -> torch.Tensor | None:
        if self._weights_exists is False:
            return None
        else:
            return self._weights

    @weights.setter
    def weights(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 2
        assert value.dtype == torch.float64
        temp: torch.Tensor = value.detach().clone(memory_format=torch.contiguous_format)
        temp /= temp.sum(dim=0, keepdim=True, dtype=torch.float64)
        if self._weights_exists is False:
            self._weights = torch.nn.parameter.Parameter(
                temp,
                requires_grad=True,
            )
            self._weights_exists = True
        else:
            self._weights.data = temp

    @property
    def kernel_size(self) -> torch.Tensor | None:
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert torch.numel(value) == 2
        assert value.dtype == torch.int64
        assert value[0] > 0
        assert value[1] > 0
        self._kernel_size = value.detach().clone(memory_format=torch.contiguous_format)
        self._kernel_size.requires_grad_(False)

    @property
    def stride(self) -> torch.Tensor | None:
        return self._stride

    @stride.setter
    def stride(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert torch.numel(value) == 2
        assert value.dtype == torch.int64
        assert value[0] > 0
        assert value[1] > 0
        self._stride = value.detach().clone(memory_format=torch.contiguous_format)
        self._stride.requires_grad_(False)

    @property
    def dilation(self) -> torch.Tensor | None:
        return self._dilation

    @dilation.setter
    def dilation(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert torch.numel(value) == 2
        assert value.dtype == torch.int64
        assert value[0] > 0
        assert value[1] > 0
        self._dilation = value.detach().clone(memory_format=torch.contiguous_format)
        self._dilation.requires_grad_(False)

    @property
    def padding(self) -> torch.Tensor | None:
        return self._padding

    @padding.setter
    def padding(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert torch.numel(value) == 2
        assert value.dtype == torch.int64
        assert value[0] >= 0
        assert value[1] >= 0
        self._padding = value.detach().clone(memory_format=torch.contiguous_format)
        self._padding.requires_grad_(False)

    @property
    def output_size(self) -> torch.Tensor | None:
        return self._output_size

    @output_size.setter
    def output_size(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert torch.numel(value) == 2
        assert value.dtype == torch.int64
        assert value[0] > 0
        assert value[1] > 0
        self._output_size = value.detach().clone(memory_format=torch.contiguous_format)
        self._output_size.requires_grad_(False)

    @property
    def number_of_spikes(self) -> torch.Tensor | None:
        return self._number_of_spikes

    @number_of_spikes.setter
    def number_of_spikes(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert torch.numel(value) == 1
        assert value.dtype == torch.int64
        assert value.item() > 0
        self._number_of_spikes = value.detach().clone(
            memory_format=torch.contiguous_format
        )
        self._number_of_spikes.requires_grad_(False)

    @property
    def number_of_cpu_processes(self) -> torch.Tensor | None:
        return self._number_of_cpu_processes

    @number_of_cpu_processes.setter
    def number_of_cpu_processes(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert torch.numel(value) == 1
        assert value.dtype == torch.int64
        assert value.item() > 0
        self._number_of_cpu_processes = value.detach().clone(
            memory_format=torch.contiguous_format
        )
        self._number_of_cpu_processes.requires_grad_(False)

    @property
    def number_of_neurons(self) -> torch.Tensor | None:
        return self._number_of_neurons

    @number_of_neurons.setter
    def number_of_neurons(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert torch.numel(value) == 1
        assert value.dtype == torch.int64
        assert value.item() > 0
        self._number_of_neurons = value.detach().clone(
            memory_format=torch.contiguous_format
        )
        self._number_of_neurons.requires_grad_(False)

    @property
    def number_of_input_neurons(self) -> torch.Tensor | None:
        return self._number_of_input_neurons

    @number_of_input_neurons.setter
    def number_of_input_neurons(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert torch.numel(value) == 1
        assert value.dtype == torch.int64
        assert value.item() > 0
        self._number_of_input_neurons = value.detach().clone(
            memory_format=torch.contiguous_format
        )
        self._number_of_input_neurons.requires_grad_(False)

    @property
    def h_initial(self) -> torch.Tensor | None:
        return self._h_initial

    @h_initial.setter
    def h_initial(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert value.dtype == torch.float32
        self._h_initial = value.detach().clone(memory_format=torch.contiguous_format)
        self._h_initial.requires_grad_(False)

    @property
    def alpha_number_of_iterations(self) -> torch.Tensor | None:
        return self._alpha_number_of_iterations

    @alpha_number_of_iterations.setter
    def alpha_number_of_iterations(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert torch.numel(value) == 1
        assert value.dtype == torch.int64
        assert value.item() >= 0
        self._alpha_number_of_iterations = value.detach().clone(
            memory_format=torch.contiguous_format
        )
        self._alpha_number_of_iterations.requires_grad_(False)

    ####################################################################
    # Forward                                                          #
    ####################################################################

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """PyTorch Forward method. Does the work."""

        # Are we happy with the input?
        assert input is not None
        assert torch.is_tensor(input) is True
        assert input.dim() == 4
        assert input.dtype == torch.float64

        # Are we happy with the rest of the network?
        assert self._epsilon_xy_exists is True
        assert self._epsilon_xy is not None
        assert self._epsilon_0 is not None
        assert self._epsilon_t is not None
        assert self._weights_exists is True
        assert self._weights is not None
        assert self._kernel_size is not None
        assert self._stride is not None
        assert self._dilation is not None
        assert self._padding is not None
        assert self._output_size is not None
        assert self._number_of_spikes is not None
        assert self._number_of_cpu_processes is not None
        assert self._h_initial is not None
        assert self._alpha_number_of_iterations is not None

        # SbS forward functional
        return self.functional_sbs(
            input,
            self._epsilon_xy,
            self._epsilon_0,
            self._epsilon_t,
            self._weights,
            self._kernel_size,
            self._stride,
            self._dilation,
            self._padding,
            self._output_size,
            self._number_of_spikes,
            self._number_of_cpu_processes,
            self._h_initial,
            self._alpha_number_of_iterations,
        )

    ####################################################################
    # Helper functions                                                 #
    ####################################################################

    def calculate_output_size(self, value: torch.Tensor) -> None:

        coordinates_0, coordinates_1 = self._get_coordinates(value)

        self._output_size: torch.Tensor = torch.tensor(
            [
                coordinates_0.shape[1],
                coordinates_1.shape[1],
            ],
            dtype=torch.int64,
        )
        self._output_size.requires_grad_(False)

    def _get_coordinates(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Function converts parameter in coordinates
        for the convolution window"""

        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert torch.numel(value) == 2
        assert value.dtype == torch.int64
        assert value[0] > 0
        assert value[1] > 0

        assert self._kernel_size is not None
        assert self._stride is not None
        assert self._dilation is not None
        assert self._padding is not None

        assert torch.numel(self._kernel_size) == 2
        assert torch.numel(self._stride) == 2
        assert torch.numel(self._dilation) == 2
        assert torch.numel(self._padding) == 2

        unfold_0: torch.nn.Unfold = torch.nn.Unfold(
            kernel_size=(int(self._kernel_size[0]), 1),
            dilation=int(self._dilation[0]),
            padding=int(self._padding[0]),
            stride=int(self._stride[0]),
        )

        unfold_1: torch.nn.Unfold = torch.nn.Unfold(
            kernel_size=(1, int(self._kernel_size[1])),
            dilation=int(self._dilation[1]),
            padding=int(self._padding[1]),
            stride=int(self._stride[1]),
        )

        coordinates_0: torch.Tensor = (
            unfold_0(
                torch.unsqueeze(
                    torch.unsqueeze(
                        torch.unsqueeze(
                            torch.arange(0, int(value[0]), dtype=torch.float64),
                            1,
                        ),
                        0,
                    ),
                    0,
                )
            )
            .squeeze(0)
            .type(torch.int64)
        )

        coordinates_1: torch.Tensor = (
            unfold_1(
                torch.unsqueeze(
                    torch.unsqueeze(
                        torch.unsqueeze(
                            torch.arange(0, int(value[1]), dtype=torch.float64),
                            0,
                        ),
                        0,
                    ),
                    0,
                )
            )
            .squeeze(0)
            .type(torch.int64)
        )

        return coordinates_0, coordinates_1

    def _initial_random_weights(self, noise_amplitude: torch.Tensor) -> torch.Tensor:
        """Creates initial weights
        Uniform plus random noise plus normalization
        """

        assert torch.numel(noise_amplitude) == 1
        assert noise_amplitude.item() >= 0
        assert noise_amplitude.dtype == torch.float64

        assert self._number_of_neurons is not None
        assert self._number_of_input_neurons is not None
        assert self._kernel_size is not None

        weights = torch.empty(
            (
                int(self._kernel_size[0]),
                int(self._kernel_size[1]),
                int(self._number_of_input_neurons),
                int(self._number_of_neurons),
            ),
            dtype=torch.float64,
        )
        torch.nn.init.uniform_(weights, a=1.0, b=(1.0 + noise_amplitude.item()))

        return weights

    def _make_pooling_weights(self) -> torch.Tensor:
        """For generating the pooling weights."""

        assert self._number_of_neurons is not None
        assert self._kernel_size is not None

        norm: float = 1.0 / (self._kernel_size[0] * self._kernel_size[1])

        weights: torch.Tensor = torch.zeros(
            (
                int(self._kernel_size[0]),
                int(self._kernel_size[1]),
                int(self._number_of_neurons),
                int(self._number_of_neurons),
            ),
            dtype=torch.float64,
        )

        for i in range(0, int(self._number_of_neurons)):
            weights[:, :, i, i] = norm

        return weights

    def initialize_weights(
        self,
        is_pooling_layer: bool = False,
        noise_amplitude: float = 0.01,
    ) -> None:
        """For the generation of the initital weights.
        Switches between normal initial random weights and pooling weights."""

        assert self._kernel_size is not None

        if is_pooling_layer is True:
            weights = self._make_pooling_weights()
        else:
            weights = self._initial_random_weights(
                torch.tensor(noise_amplitude, dtype=torch.float64)
            )

        weights = weights.moveaxis(-1, 0).moveaxis(-1, 1)

        weights_t = torch.nn.functional.unfold(
            input=weights,
            kernel_size=(int(self._kernel_size[0]), int(self._kernel_size[1])),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        ).squeeze()

        weights_t = torch.moveaxis(weights_t, 0, 1)

        self.weights = weights_t

    def initialize_epsilon_xy(
        self,
        eps_xy_intitial: float,
    ) -> None:
        """Creates initial epsilon xy matrices"""

        assert self._output_size is not None
        assert self._kernel_size is not None
        assert eps_xy_intitial > 0

        eps_xy_temp: torch.Tensor = torch.full(
            (
                int(self._output_size[0]),
                int(self._output_size[1]),
                int(self._kernel_size[0]),
                int(self._kernel_size[1]),
            ),
            eps_xy_intitial,
            dtype=torch.float64,
        )

        self.epsilon_xy = eps_xy_temp

    def set_h_init_to_uniform(self) -> None:

        assert self._number_of_neurons is not None

        h_initial: torch.Tensor = torch.full(
            (int(self._number_of_neurons.item()),),
            (1.0 / float(self._number_of_neurons.item())),
            dtype=torch.float32,
        )

        self.h_initial = h_initial

    # Epsilon XY
    def backup_epsilon_xy(self) -> None:
        assert self._epsilon_xy_exists is True
        self._epsilon_xy_backup = self._epsilon_xy.data.clone()

    def restore_epsilon_xy(self) -> None:
        assert self._epsilon_xy_backup is not None
        assert self._epsilon_xy_exists is True
        self._epsilon_xy.data = self._epsilon_xy_backup.clone()

    def mean_epsilon_xy(self) -> None:
        assert self._epsilon_xy_exists is True

        fill_value: float = float(self._epsilon_xy.data.mean())
        self._epsilon_xy.data = torch.full_like(
            self._epsilon_xy.data, fill_value, dtype=torch.float64
        )

    def threshold_epsilon_xy(self, threshold: float) -> None:
        assert self._epsilon_xy_exists is True
        assert threshold >= 0
        torch.clamp(
            self._epsilon_xy.data,
            min=float(threshold),
            max=None,
            out=self._epsilon_xy.data,
        )

    # Weights
    def backup_weights(self) -> None:
        assert self._weights_exists is True
        self._weights_backup = self._weights.data.clone()

    def restore_weights(self) -> None:
        assert self._weights_backup is not None
        assert self._weights_exists is True
        self._weights.data = self._weights_backup.clone()

    def norm_weights(self) -> None:
        assert self._weights_exists is True
        temp: torch.Tensor = (
            self._weights.data.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=torch.float64)
        )
        temp /= temp.sum(dim=0, keepdim=True, dtype=torch.float64)
        self._weights.data = temp

    def threshold_weights(self, threshold: float) -> None:
        assert self._weights_exists is True
        assert threshold >= 0
        torch.clamp(
            self._weights.data,
            min=float(threshold),
            max=None,
            out=self._weights.data,
        )


class FunctionalSbS(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input_float64: torch.Tensor,
        epsilon_xy_float64: torch.Tensor,
        epsilon_0_float64: torch.Tensor,
        epsilon_t_float64: torch.Tensor,
        weights_float64: torch.Tensor,
        kernel_size: torch.Tensor,
        stride: torch.Tensor,
        dilation: torch.Tensor,
        padding: torch.Tensor,
        output_size: torch.Tensor,
        number_of_spikes: torch.Tensor,
        number_of_cpu_processes: torch.Tensor,
        h_initial: torch.Tensor,
        alpha_number_of_iterations: torch.Tensor,
    ) -> torch.Tensor:

        input = input_float64.type(dtype=torch.float32)
        epsilon_xy = epsilon_xy_float64.type(dtype=torch.float32)
        weights = weights_float64.type(dtype=torch.float32)
        epsilon_0 = epsilon_0_float64.type(dtype=torch.float32)
        epsilon_t = epsilon_t_float64.type(dtype=torch.float32)

        assert input.dim() == 4
        assert torch.numel(kernel_size) == 2
        assert torch.numel(dilation) == 2
        assert torch.numel(padding) == 2
        assert torch.numel(stride) == 2
        assert torch.numel(output_size) == 2

        assert torch.numel(epsilon_0) == 1
        assert torch.numel(number_of_spikes) == 1
        assert torch.numel(number_of_cpu_processes) == 1
        assert torch.numel(alpha_number_of_iterations) == 1

        input_size = torch.tensor([input.shape[2], input.shape[3]])

        ############################################################
        # Pre convolving the input                                 #
        ############################################################

        input_convolved_temp = torch.nn.functional.unfold(
            input,
            kernel_size=tuple(kernel_size.tolist()),
            dilation=tuple(dilation.tolist()),
            padding=tuple(padding.tolist()),
            stride=tuple(stride.tolist()),
        )

        input_convolved = torch.nn.functional.fold(
            input_convolved_temp,
            output_size=tuple(output_size.tolist()),
            kernel_size=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        ).requires_grad_(True)

        epsilon_xy_convolved: torch.Tensor = (
            (
                torch.nn.functional.unfold(
                    epsilon_xy.reshape(
                        (
                            int(epsilon_xy.shape[0]) * int(epsilon_xy.shape[1]),
                            int(epsilon_xy.shape[2]),
                            int(epsilon_xy.shape[3]),
                        )
                    )
                    .unsqueeze(1)
                    .tile((1, input.shape[1], 1, 1)),
                    kernel_size=tuple(kernel_size.tolist()),
                    dilation=1,
                    padding=0,
                    stride=1,
                )
                .squeeze(-1)
                .reshape(
                    (
                        int(epsilon_xy.shape[0]),
                        int(epsilon_xy.shape[1]),
                        int(input_convolved.shape[1]),
                    )
                )
            )
            .moveaxis(-1, 0)
            .contiguous(memory_format=torch.contiguous_format)
        )

        ############################################################
        # Spike generation                                         #
        ############################################################

        if cpp_spike is False:
            # Alternative to the C++ module but 5x slower:
            spikes = (
                (
                    input_convolved.movedim(source=(2, 3), destination=(0, 1))
                    .reshape(
                        shape=(
                            input_convolved.shape[2]
                            * input_convolved.shape[3]
                            * input_convolved.shape[0],
                            input_convolved.shape[1],
                        )
                    )
                    .multinomial(
                        num_samples=int(number_of_spikes.item()), replacement=True
                    )
                )
                .reshape(
                    shape=(
                        input_convolved.shape[2],
                        input_convolved.shape[3],
                        input_convolved.shape[0],
                        int(number_of_spikes.item()),
                    )
                )
                .movedim(source=(0, 1), destination=(2, 3))
            ).contiguous(memory_format=torch.contiguous_format)
        else:
            # Normalized cumsum
            input_cumsum: torch.Tensor = torch.cumsum(
                input_convolved, dim=1, dtype=torch.float32
            )
            input_cumsum_last: torch.Tensor = input_cumsum[:, -1, :, :].unsqueeze(1)
            input_cumsum /= input_cumsum_last

            random_values = torch.rand(
                size=[
                    input_cumsum.shape[0],
                    int(number_of_spikes.item()),
                    input_cumsum.shape[2],
                    input_cumsum.shape[3],
                ],
                dtype=torch.float32,
            )

            spikes = torch.empty_like(random_values, dtype=torch.int64)

            # Prepare for Export (Pointer and stuff)->
            np_input: np.ndarray = input_cumsum.detach().numpy()
            assert input_cumsum.dtype == torch.float32
            assert np_input.flags["C_CONTIGUOUS"] is True
            assert np_input.ndim == 4

            np_random_values: np.ndarray = random_values.detach().numpy()
            assert random_values.dtype == torch.float32
            assert np_random_values.flags["C_CONTIGUOUS"] is True
            assert np_random_values.ndim == 4

            np_spikes: np.ndarray = spikes.detach().numpy()
            assert spikes.dtype == torch.int64
            assert np_spikes.flags["C_CONTIGUOUS"] is True
            assert np_spikes.ndim == 4
            # <- Prepare for Export

            spike_generation: PySpikeGeneration2DManyIP.SpikeGeneration2DManyIP = (
                PySpikeGeneration2DManyIP.SpikeGeneration2DManyIP()
            )

            spike_generation.spike_generation_multi_pattern(
                np_input.__array_interface__["data"][0],
                int(np_input.shape[0]),
                int(np_input.shape[1]),
                int(np_input.shape[2]),
                int(np_input.shape[3]),
                np_random_values.__array_interface__["data"][0],
                int(np_random_values.shape[0]),
                int(np_random_values.shape[1]),
                int(np_random_values.shape[2]),
                int(np_random_values.shape[3]),
                np_spikes.__array_interface__["data"][0],
                int(np_spikes.shape[0]),
                int(np_spikes.shape[1]),
                int(np_spikes.shape[2]),
                int(np_spikes.shape[3]),
                int(number_of_cpu_processes.item()),
            )

        ############################################################
        # H dynamic                                                #
        ############################################################

        assert epsilon_t.ndim == 1
        assert epsilon_t.shape[0] >= number_of_spikes

        if cpp_sbs is False:
            h = torch.tile(
                h_initial.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                dims=[int(input.shape[0]), int(output_size[0]), int(output_size[1]), 1],
            )

            epsilon_scale: torch.Tensor = torch.ones(
                size=[
                    int(spikes.shape[0]),
                    int(spikes.shape[2]),
                    int(spikes.shape[3]),
                    1,
                ],
                dtype=torch.float32,
            )

            for t in range(0, spikes.shape[1]):

                if epsilon_scale.max() > 1e10:
                    h /= epsilon_scale
                    epsilon_scale = torch.ones_like(epsilon_scale)

                h_temp: torch.Tensor = weights[spikes[:, t, :, :], :] * h
                wx = 0
                wy = 0

                if t == 0:
                    epsilon_temp: torch.Tensor = torch.empty(
                        (
                            int(spikes.shape[0]),
                            int(spikes.shape[2]),
                            int(spikes.shape[3]),
                        ),
                        dtype=torch.float32,
                    )
                for wx in range(0, int(spikes.shape[2])):
                    for wy in range(0, int(spikes.shape[3])):
                        epsilon_temp[:, wx, wy] = epsilon_xy_convolved[
                            spikes[:, t, wx, wy], wx, wy
                        ]

                epsilon_subsegment: torch.Tensor = (
                    epsilon_temp.unsqueeze(-1) * epsilon_t[t] * epsilon_0
                )

                h_temp_sum: torch.Tensor = (
                    epsilon_scale * epsilon_subsegment / h_temp.sum(dim=3, keepdim=True)
                )
                torch.nan_to_num(
                    h_temp_sum, out=h_temp_sum, nan=0.0, posinf=0.0, neginf=0.0
                )
                h_temp *= h_temp_sum
                h += h_temp

                epsilon_scale *= 1.0 + epsilon_subsegment

            h /= epsilon_scale
            output = h.movedim(3, 1)

        else:
            epsilon_t_0: torch.Tensor = epsilon_t * epsilon_0

            h_shape: tuple[int, int, int, int] = (
                int(input.shape[0]),
                int(weights.shape[1]),
                int(output_size[0]),
                int(output_size[1]),
            )

            output = torch.empty(h_shape, dtype=torch.float32)

            # Prepare the export to C++ ->
            np_h: np.ndarray = output.detach().numpy()
            assert output.dtype == torch.float32
            assert np_h.flags["C_CONTIGUOUS"] is True
            assert np_h.ndim == 4

            np_epsilon_xy: np.ndarray = epsilon_xy_convolved.detach().numpy()
            assert epsilon_xy.dtype == torch.float32
            assert np_epsilon_xy.flags["C_CONTIGUOUS"] is True
            assert np_epsilon_xy.ndim == 3

            np_epsilon_t: np.ndarray = epsilon_t_0.detach().numpy()
            assert epsilon_t_0.dtype == torch.float32
            assert np_epsilon_t.flags["C_CONTIGUOUS"] is True
            assert np_epsilon_t.ndim == 1

            np_weights: np.ndarray = weights.detach().numpy()
            assert weights.dtype == torch.float32
            assert np_weights.flags["C_CONTIGUOUS"] is True
            assert np_weights.ndim == 2

            np_spikes = spikes.contiguous().detach().numpy()
            assert spikes.dtype == torch.int64
            assert np_spikes.flags["C_CONTIGUOUS"] is True
            assert np_spikes.ndim == 4

            np_h_initial = h_initial.contiguous().detach().numpy()
            assert h_initial.dtype == torch.float32
            assert np_h_initial.flags["C_CONTIGUOUS"] is True
            assert np_h_initial.ndim == 1
            # <- Prepare the export to C++

            h_dynamic: PyHDynamicCNNManyIP.HDynamicCNNManyIP = (
                PyHDynamicCNNManyIP.HDynamicCNNManyIP()
            )

            h_dynamic.update_with_init_vector_multi_pattern(
                np_h.__array_interface__["data"][0],
                int(np_h.shape[0]),
                int(np_h.shape[1]),
                int(np_h.shape[2]),
                int(np_h.shape[3]),
                np_epsilon_xy.__array_interface__["data"][0],
                int(np_epsilon_xy.shape[0]),
                int(np_epsilon_xy.shape[1]),
                int(np_epsilon_xy.shape[2]),
                np_epsilon_t.__array_interface__["data"][0],
                int(np_epsilon_t.shape[0]),
                np_weights.__array_interface__["data"][0],
                int(np_weights.shape[0]),
                int(np_weights.shape[1]),
                np_spikes.__array_interface__["data"][0],
                int(np_spikes.shape[0]),
                int(np_spikes.shape[1]),
                int(np_spikes.shape[2]),
                int(np_spikes.shape[3]),
                np_h_initial.__array_interface__["data"][0],
                int(np_h_initial.shape[0]),
                int(number_of_cpu_processes.item()),
            )

        ############################################################
        # Alpha                                                    #
        ############################################################
        alpha_number_of_iterations_int: int = int(alpha_number_of_iterations.item())

        if alpha_number_of_iterations_int > 0:
            # Initialization
            virtual_reconstruction_weight: torch.Tensor = torch.einsum(
                "bixy,ji->bjxy", output, weights
            )
            alpha_fill_value: float = 1.0 / (
                virtual_reconstruction_weight.shape[2]
                * virtual_reconstruction_weight.shape[3]
            )
            alpha_dynamic: torch.Tensor = torch.full(
                (
                    int(virtual_reconstruction_weight.shape[0]),
                    1,
                    int(virtual_reconstruction_weight.shape[2]),
                    int(virtual_reconstruction_weight.shape[3]),
                ),
                alpha_fill_value,
                dtype=torch.float32,
            )

            # Iterations
            for _ in range(0, alpha_number_of_iterations_int):
                alpha_temp: torch.Tensor = alpha_dynamic * virtual_reconstruction_weight
                alpha_temp /= alpha_temp.sum(dim=3, keepdim=True).sum(
                    dim=2, keepdim=True
                )
                torch.nan_to_num(
                    alpha_temp, out=alpha_temp, nan=0.0, posinf=0.0, neginf=0.0
                )

                alpha_temp = torch.nn.functional.unfold(
                    alpha_temp,
                    kernel_size=(1, 1),
                    dilation=1,
                    padding=0,
                    stride=1,
                )

                alpha_temp = torch.nn.functional.fold(
                    alpha_temp,
                    output_size=tuple(input_size.tolist()),
                    kernel_size=tuple(kernel_size.tolist()),
                    dilation=tuple(dilation.tolist()),
                    padding=tuple(padding.tolist()),
                    stride=tuple(stride.tolist()),
                )

                alpha_temp = (alpha_temp * input).sum(dim=1, keepdim=True)

                alpha_temp = torch.nn.functional.unfold(
                    alpha_temp,
                    kernel_size=tuple(kernel_size.tolist()),
                    dilation=tuple(dilation.tolist()),
                    padding=tuple(padding.tolist()),
                    stride=tuple(stride.tolist()),
                )

                alpha_temp = torch.nn.functional.fold(
                    alpha_temp,
                    output_size=tuple(output_size.tolist()),
                    kernel_size=(1, 1),
                    dilation=(1, 1),
                    padding=(0, 0),
                    stride=(1, 1),
                )
                alpha_dynamic = alpha_temp.sum(dim=1, keepdim=True)

                alpha_dynamic += torch.finfo(torch.float32).eps * 1000

                # Alpha normalization
                alpha_dynamic /= alpha_dynamic.sum(dim=3, keepdim=True).sum(
                    dim=2, keepdim=True
                )
                torch.nan_to_num(
                    alpha_dynamic, out=alpha_dynamic, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Applied to the output
            output *= alpha_dynamic

        ############################################################
        # Save the necessary data for the backward pass            #
        ############################################################

        output = output.type(dtype=torch.float64)

        ctx.save_for_backward(
            input_convolved,
            epsilon_xy_convolved,
            epsilon_0_float64,
            weights_float64,
            output,
            kernel_size,
            stride,
            dilation,
            padding,
            input_size,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Get the variables back
        (
            input_float32,
            epsilon_xy_float32,
            epsilon_0,
            weights,
            output,
            kernel_size,
            stride,
            dilation,
            padding,
            input_size,
        ) = ctx.saved_tensors

        input = input_float32.type(dtype=torch.float64)
        input /= input.sum(dim=1, keepdim=True, dtype=torch.float64)
        epsilon_xy = epsilon_xy_float32.type(dtype=torch.float64)

        # For debugging:
        # print(
        #     f"S: O: {output.min().item():e} {output.max().item():e} I: {input.min().item():e} {input.max().item():e} G: {grad_output.min().item():e} {grad_output.max().item():e}"
        # )

        epsilon_0_float: float = epsilon_0.item()

        temp_e: torch.Tensor = 1.0 / ((epsilon_xy * epsilon_0_float) + 1.0)

        eps_a: torch.Tensor = temp_e.clone()
        eps_a *= epsilon_xy * epsilon_0_float

        eps_b: torch.Tensor = temp_e**2 * epsilon_0_float

        backprop_r: torch.Tensor = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1
        ) * output.unsqueeze(1)

        backprop_bigr: torch.Tensor = backprop_r.sum(axis=2)

        temp: torch.Tensor = input / backprop_bigr**2

        backprop_f: torch.Tensor = output.unsqueeze(1) * temp.unsqueeze(2)
        torch.nan_to_num(
            backprop_f, out=backprop_f, nan=1e300, posinf=1e300, neginf=-1e300
        )
        torch.clip(backprop_f, out=backprop_f, min=-1e300, max=1e300)

        tempz: torch.Tensor = 1.0 / backprop_bigr

        backprop_z: torch.Tensor = backprop_r * tempz.unsqueeze(2)
        torch.nan_to_num(
            backprop_z, out=backprop_z, nan=1e300, posinf=1e300, neginf=-1e300
        )
        torch.clip(backprop_z, out=backprop_z, min=-1e300, max=1e300)

        result_omega: torch.Tensor = backprop_bigr.unsqueeze(2) * grad_output.unsqueeze(
            1
        )
        result_omega -= torch.einsum(
            "bijxy,bjxy->bixy", backprop_r, grad_output
        ).unsqueeze(2)
        result_omega *= backprop_f

        result_eps_xy: torch.Tensor = (
            (
                (backprop_z * input.unsqueeze(2) - output.unsqueeze(1))
                * grad_output.unsqueeze(1)
            )
            .sum(dim=2)
            .sum(dim=0)
        ) * eps_b

        result_phi: torch.Tensor = torch.einsum(
            "bijxy,bjxy->bixy", backprop_z, grad_output
        ) * eps_a.unsqueeze(0)

        grad_weights = result_omega.sum(0).sum(-1).sum(-1)
        torch.nan_to_num(
            grad_weights, out=grad_weights, nan=1e300, posinf=1e300, neginf=-1e300
        )
        torch.clip(grad_weights, out=grad_weights, min=-1e300, max=1e300)

        grad_input = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                result_phi,
                kernel_size=(1, 1),
                dilation=1,
                padding=0,
                stride=1,
            ),
            output_size=input_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        torch.nan_to_num(
            grad_input, out=grad_input, nan=1e300, posinf=1e300, neginf=-1e300
        )
        torch.clip(grad_input, out=grad_input, min=-1e300, max=1e300)

        grad_eps_xy_temp = torch.nn.functional.fold(
            result_eps_xy.moveaxis(0, -1)
            .reshape(
                (
                    int(result_eps_xy.shape[1]) * int(result_eps_xy.shape[2]),
                    int(result_eps_xy.shape[0]),
                )
            )
            .unsqueeze(-1),
            output_size=kernel_size,
            kernel_size=kernel_size,
        )

        grad_eps_xy = (
            grad_eps_xy_temp.sum(dim=1)
            .reshape(
                (
                    int(result_eps_xy.shape[1]),
                    int(result_eps_xy.shape[2]),
                    int(grad_eps_xy_temp.shape[-2]),
                    int(grad_eps_xy_temp.shape[-1]),
                )
            )
            .contiguous(memory_format=torch.contiguous_format)
        )
        torch.nan_to_num(
            grad_eps_xy, out=grad_eps_xy, nan=1e300, posinf=1e300, neginf=-1e300
        )
        torch.clip(grad_eps_xy, out=grad_eps_xy, min=-1e300, max=1e300)

        grad_epsilon_0 = None
        grad_epsilon_t = None
        grad_kernel_size = None
        grad_stride = None
        grad_dilation = None
        grad_padding = None
        grad_output_size = None
        grad_number_of_spikes = None
        grad_number_of_cpu_processes = None
        grad_h_initial = None
        grad_alpha_number_of_iterations = None

        return (
            grad_input,
            grad_eps_xy,
            grad_epsilon_0,
            grad_epsilon_t,
            grad_weights,
            grad_kernel_size,
            grad_stride,
            grad_dilation,
            grad_padding,
            grad_output_size,
            grad_number_of_spikes,
            grad_number_of_cpu_processes,
            grad_h_initial,
            grad_alpha_number_of_iterations,
        )


# %%
