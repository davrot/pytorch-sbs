import torch

from network.PySpikeGenerationCPU import SpikeGenerationCPU
from network.PySpikeGenerationGPU import SpikeGenerationGPU

from network.PyHDynamicCNNCPU import HDynamicCNNCPU
from network.PyHDynamicCNNGPU import HDynamicCNNGPU

from network.calculate_output_size import calculate_output_size
import os
import numpy as np

global_sbs_gpu_setting: list[torch.Tensor] = []
global_sbs_size: list[torch.Tensor] = []
global_sbs_hdynamic_cpp: list[HDynamicCNNCPU | HDynamicCNNGPU] = []
global_spike_generation_gpu_setting: list[torch.Tensor] = []
global_spike_size: list[torch.Tensor] = []
global_spike_generation_cpp: list[SpikeGenerationCPU | SpikeGenerationGPU] = []


class SbS(torch.nn.Module):

    _epsilon_xy: torch.Tensor | None = None
    _epsilon_0: float
    _weights: torch.nn.parameter.Parameter
    _weights_exists: bool = False
    _kernel_size: list[int]
    _stride: list[int]
    _dilation: list[int]
    _padding: list[int]
    _output_size: torch.Tensor
    _number_of_spikes: int
    _number_of_cpu_processes: int
    _number_of_neurons: int
    _number_of_input_neurons: int
    _epsilon_xy_intitial: float
    _h_initial: torch.Tensor | None = None
    _w_trainable: bool
    _last_grad_scale: torch.nn.parameter.Parameter
    _keep_last_grad_scale: bool
    _disable_scale_grade: bool
    _forgetting_offset: torch.Tensor | None = None
    _weight_noise_range: list[float]
    _skip_gradient_calculation: bool
    _is_pooling_layer: bool
    _input_size: list[int]
    _output_layer: bool = False
    _local_learning: bool = False

    device: torch.device
    default_dtype: torch.dtype
    _gpu_tuning_factor: int

    _max_grad_weights: torch.Tensor | None = None

    _number_of_grad_weight_contributions: float = 0.0

    last_input_store: bool = False
    last_input_data: torch.Tensor | None = None

    sbs_gpu_setting_position: int = -1
    sbs_hdynamic_cpp_position: int = -1
    spike_generation_cpp_position: int = -1
    spike_generation_gpu_setting_position: int = -1

    _cooldown_after_number_of_spikes: int = -1
    _reduction_cooldown: float = 1.0
    _layer_id: int = (-1,)

    def __init__(
        self,
        number_of_input_neurons: int,
        number_of_neurons: int,
        input_size: list[int],
        forward_kernel_size: list[int],
        number_of_spikes: int,
        epsilon_xy_intitial: float = 0.1,
        epsilon_0: float = 1.0,
        weight_noise_range: list[float] = [0.0, 1.0],
        is_pooling_layer: bool = False,
        strides: list[int] = [1, 1],
        dilation: list[int] = [0, 0],
        padding: list[int] = [0, 0],
        number_of_cpu_processes: int = 1,
        w_trainable: bool = False,
        keep_last_grad_scale: bool = False,
        disable_scale_grade: bool = True,
        forgetting_offset: float = -1.0,
        skip_gradient_calculation: bool = False,
        device: torch.device | None = None,
        default_dtype: torch.dtype | None = None,
        gpu_tuning_factor: int = 5,
        layer_id: int = -1,
        cooldown_after_number_of_spikes: int = -1,
        reduction_cooldown: float = 1.0,
    ) -> None:
        super().__init__()

        assert device is not None
        assert default_dtype is not None
        self.device = device
        self.default_dtype = default_dtype

        self._w_trainable = bool(w_trainable)
        self._keep_last_grad_scale = bool(keep_last_grad_scale)
        self._skip_gradient_calculation = bool(skip_gradient_calculation)
        self._disable_scale_grade = bool(disable_scale_grade)
        self._epsilon_xy_intitial = float(epsilon_xy_intitial)
        self._stride = strides
        self._dilation = dilation
        self._padding = padding
        self._kernel_size = forward_kernel_size
        self._number_of_input_neurons = int(number_of_input_neurons)
        self._number_of_neurons = int(number_of_neurons)
        self._epsilon_0 = float(epsilon_0)
        self._number_of_cpu_processes = int(number_of_cpu_processes)
        self._number_of_spikes = int(number_of_spikes)
        self._weight_noise_range = weight_noise_range
        self._is_pooling_layer = bool(is_pooling_layer)
        self._cooldown_after_number_of_spikes = int(cooldown_after_number_of_spikes)
        self.reduction_cooldown = float(reduction_cooldown)
        self._layer_id = layer_id

        assert len(input_size) == 2
        self._input_size = input_size

        global_sbs_gpu_setting.append(torch.tensor([0]))
        global_spike_generation_gpu_setting.append(torch.tensor([0]))
        global_sbs_size.append(torch.tensor([0, 0, 0, 0]))
        global_spike_size.append(torch.tensor([0, 0, 0, 0]))

        if device == torch.device("cpu"):
            global_sbs_hdynamic_cpp.append(HDynamicCNNGPU())
        else:
            global_sbs_hdynamic_cpp.append(HDynamicCNNCPU())

        if device == torch.device("cpu"):
            global_spike_generation_cpp.append(SpikeGenerationCPU())
        else:
            global_spike_generation_cpp.append(SpikeGenerationGPU())

        self.sbs_gpu_setting_position = len(global_sbs_gpu_setting) - 1
        self.sbs_hdynamic_cpp_position = len(global_sbs_hdynamic_cpp) - 1
        self.spike_generation_cpp_position = len(global_spike_generation_cpp) - 1
        self.spike_generation_gpu_setting_position = (
            len(global_spike_generation_gpu_setting) - 1
        )

        # The GPU hates me...
        # Too many SbS threads == bad
        # Thus I need to limit them...
        # (Reminder: We cannot access the mini-batch size here,
        # which is part of the GPU thread size calculation...)
        if (self._input_size[0] * self._input_size[1]) > gpu_tuning_factor:
            self._gpu_tuning_factor = gpu_tuning_factor
        else:
            self._gpu_tuning_factor = 0

        self._last_grad_scale = torch.nn.parameter.Parameter(
            torch.tensor(-1.0, dtype=self.default_dtype),
            requires_grad=True,
        )

        self._forgetting_offset = torch.tensor(
            forgetting_offset, dtype=self.default_dtype, device=self.device
        )

        self._output_size = calculate_output_size(
            value=input_size,
            kernel_size=self._kernel_size,
            stride=self._stride,
            dilation=self._dilation,
            padding=self._padding,
        )

        self.set_h_init_to_uniform()

        self.functional_sbs = FunctionalSbS.apply
        self.functional_spike_generation = FunctionalSpikeGeneration.apply

        # ###############################################################
        # Initialize the weights
        # ###############################################################

        if self._is_pooling_layer is True:
            self.weights = self._make_pooling_weights()

        else:
            assert len(self._weight_noise_range) == 2
            weights = torch.empty(
                (
                    int(self._kernel_size[0])
                    * int(self._kernel_size[1])
                    * int(self._number_of_input_neurons),
                    int(self._number_of_neurons),
                ),
                dtype=self.default_dtype,
                device=self.device,
            )

            torch.nn.init.uniform_(
                weights,
                a=float(self._weight_noise_range[0]),
                b=float(self._weight_noise_range[1]),
            )
            self.weights = weights

    ####################################################################
    # Variables in and out                                             #
    ####################################################################

    def get_epsilon_t(self, number_of_spikes: int):
        """Generates the time series of the basic epsilon."""
        t = np.arange(0, number_of_spikes, dtype=np.float32) + 1
        np_epsilon_t: np.ndarray = t ** (
            -1.0 / 2.0
        )  # np.ones((number_of_spikes), dtype=np.float32)

        if (self._cooldown_after_number_of_spikes < number_of_spikes) and (
            self._cooldown_after_number_of_spikes >= 0
        ):
            np_epsilon_t[
                self._cooldown_after_number_of_spikes : number_of_spikes
            ] /= self._reduction_cooldown
        return (
            torch.tensor(np_epsilon_t)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
        )

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
        temp: torch.Tensor = (
            value.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
        )
        temp /= temp.sum(dim=0, keepdim=True, dtype=self.default_dtype)
        if self._weights_exists is False:
            self._weights = torch.nn.parameter.Parameter(temp, requires_grad=True)
            self._weights_exists = True
        else:
            self._weights.data = temp

    @property
    def h_initial(self) -> torch.Tensor | None:
        return self._h_initial

    @h_initial.setter
    def h_initial(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert value.dtype == self.default_dtype
        self._h_initial = (
            value.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
            .requires_grad_(False)
        )

    def update_pre_care(self):

        if self._weights.grad is not None:
            assert self._number_of_grad_weight_contributions > 0
            self._weights.grad /= self._number_of_grad_weight_contributions
            self._number_of_grad_weight_contributions = 0.0

    def update_after_care(self, threshold_weight: float):

        if self._w_trainable is True:
            self.norm_weights()
            self.threshold_weights(threshold_weight)
            self.norm_weights()

    def after_batch(self, new_state: bool = False):
        if self._keep_last_grad_scale is True:
            self._last_grad_scale.data = self._last_grad_scale.grad
            self._keep_last_grad_scale = new_state

        self._last_grad_scale.grad = torch.zeros_like(self._last_grad_scale.grad)

    ####################################################################
    # Helper functions                                                 #
    ####################################################################

    def _make_pooling_weights(self) -> torch.Tensor:
        """For generating the pooling weights."""

        assert self._number_of_neurons is not None
        assert self._kernel_size is not None

        weights: torch.Tensor = torch.zeros(
            (
                int(self._kernel_size[0]),
                int(self._kernel_size[1]),
                int(self._number_of_neurons),
                int(self._number_of_neurons),
            ),
            dtype=self.default_dtype,
            device=self.device,
        )

        for i in range(0, int(self._number_of_neurons)):
            weights[:, :, i, i] = 1.0

        weights = weights.moveaxis(-1, 0).moveaxis(-1, 1)

        weights = torch.nn.functional.unfold(
            input=weights,
            kernel_size=(int(self._kernel_size[0]), int(self._kernel_size[1])),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        ).squeeze()

        weights = torch.moveaxis(weights, 0, 1)

        return weights

    def set_h_init_to_uniform(self) -> None:

        assert self._number_of_neurons > 2

        self.h_initial: torch.Tensor = torch.full(
            (self._number_of_neurons,),
            (1.0 / float(self._number_of_neurons)),
            dtype=self.default_dtype,
            device=self.device,
        )

    def norm_weights(self) -> None:
        assert self._weights_exists is True
        temp: torch.Tensor = (
            self._weights.data.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
        )
        temp /= temp.sum(dim=0, keepdim=True, dtype=self.default_dtype)
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

    ####################################################################
    # Forward                                                          #
    ####################################################################

    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor | None = None,
        extract_noisy_pictures: bool = False,
        layer_id: int = -1,
        mini_batch_id: int = -1,
        overwrite_number_of_spikes: int = -1,
    ) -> torch.Tensor:

        # Are we happy with the input?
        assert input is not None
        assert torch.is_tensor(input) is True
        assert input.dim() == 4
        assert input.dtype == self.default_dtype
        assert input.shape[1] == self._number_of_input_neurons
        assert input.shape[2] == self._input_size[0]
        assert input.shape[3] == self._input_size[1]

        # Are we happy with the rest of the network?
        assert self._epsilon_0 is not None

        assert self._h_initial is not None
        assert self._forgetting_offset is not None

        assert self._weights_exists is True
        assert self._weights is not None

        assert self.sbs_gpu_setting_position != -1
        assert self.sbs_hdynamic_cpp_position != -1
        assert self.spike_generation_cpp_position != -1
        assert self.spike_generation_gpu_setting_position != -1

        if labels is None:
            labels_copy: torch.Tensor = torch.tensor(
                [], dtype=torch.int64, device=self.device
            )
        else:
            labels_copy = (
                labels.detach().clone().type(dtype=torch.int64).to(device=self.device)
            )

        input_convolved = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                input.requires_grad_(True),
                kernel_size=(int(self._kernel_size[0]), int(self._kernel_size[1])),
                dilation=(int(self._dilation[0]), int(self._dilation[1])),
                padding=(int(self._padding[0]), int(self._padding[1])),
                stride=(int(self._stride[0]), int(self._stride[1])),
            ),
            output_size=tuple(self._output_size.tolist()),
            kernel_size=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )

        if self.last_input_store is True:
            self.last_input_data = input_convolved.detach().clone()
            self.last_input_data /= self.last_input_data.sum(dim=1, keepdim=True)
        else:
            self.last_input_data = None

        if overwrite_number_of_spikes >= 1:
            _number_of_spikes = int(overwrite_number_of_spikes)
        else:
            _number_of_spikes = int(self._number_of_spikes)

        epsilon_t_0: torch.Tensor = (
            (self.get_epsilon_t(_number_of_spikes) * self._epsilon_0)
            .type(input.dtype)
            .to(input.device)
        )

        parameter_list = torch.tensor(
            [
                int(self._w_trainable),  # 0
                int(self._disable_scale_grade),  # 1
                int(self._keep_last_grad_scale),  # 2
                int(self._skip_gradient_calculation),  # 3
                int(_number_of_spikes),  # 4
                int(self._number_of_cpu_processes),  # 5
                int(self._output_size[0]),  # 6
                int(self._output_size[1]),  # 7
                int(self._gpu_tuning_factor),  # 8
                int(self._output_layer),  # 9
                int(self._local_learning),  # 10
                int(self.sbs_gpu_setting_position),  # 11
                int(self.sbs_hdynamic_cpp_position),  # 12
                int(self.spike_generation_cpp_position),  # 13
                int(self.spike_generation_gpu_setting_position),  # 14
            ],
            dtype=torch.int64,
        )

        if self._epsilon_xy is None:
            self._epsilon_xy = torch.full(
                (
                    input_convolved.shape[1],
                    input_convolved.shape[2],
                    input_convolved.shape[3],
                ),
                float(self._epsilon_xy_intitial),
                dtype=self.default_dtype,
                device=self.device,
            )

        assert self._epsilon_xy is not None
        # In the case somebody tried to replace the matrix with wrong dimensions
        assert self._epsilon_xy.shape[0] == input_convolved.shape[1]
        assert self._epsilon_xy.shape[1] == input_convolved.shape[2]
        assert self._epsilon_xy.shape[2] == input_convolved.shape[3]

        spike = self.functional_spike_generation(input_convolved, parameter_list)

        if (
            (extract_noisy_pictures is True)
            and (layer_id == 0)
            and (labels is not None)
            and (mini_batch_id >= 0)
        ):
            assert labels.shape[0] == spike.shape[0]

            path_sub: str = "noisy_picture_data"
            path_sub_spikes: str = f"{int(_number_of_spikes)}"
            path = os.path.join(path_sub, path_sub_spikes)
            os.makedirs(path_sub, exist_ok=True)
            os.makedirs(path, exist_ok=True)

            the_images = torch.zeros_like(
                input_convolved, dtype=torch.int64, device=self.device
            )

            for p_id in range(0, the_images.shape[0]):
                for sp_id in range(0, spike.shape[1]):
                    for x_id in range(0, the_images.shape[2]):
                        for y_id in range(0, the_images.shape[3]):
                            the_images[
                                p_id, spike[p_id, sp_id, x_id, y_id], x_id, y_id
                            ] += 1

            np.savez_compressed(
                os.path.join(path, f"{mini_batch_id}.npz"),
                the_images=the_images.cpu().numpy(),
                labels=labels.cpu().numpy(),
            )

        assert spike.shape[1] == _number_of_spikes

        # SbS forward functional
        output = self.functional_sbs(
            input_convolved,
            spike,
            self._epsilon_xy,
            epsilon_t_0,
            self._weights,
            self._h_initial,
            parameter_list,
            self._last_grad_scale,
            self._forgetting_offset,
            labels_copy,
        )

        self._number_of_grad_weight_contributions += (
            output.shape[0] * output.shape[-2] * output.shape[-1]
        )

        return output


class FunctionalSpikeGeneration(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        parameter_list: torch.Tensor,
    ) -> torch.Tensor:

        assert input.dim() == 4

        number_of_spikes: int = int(parameter_list[4])

        if input.device == torch.device("cpu"):
            spike_number_of_cpu_processes: int = int(parameter_list[5])
        else:
            spike_number_of_cpu_processes = -1

        spike_generation_cpp_position = int(parameter_list[13])
        spike_generation_gpu_setting_position = int(parameter_list[14])

        # ###########################################################
        # Spike generation
        # ###########################################################

        # ############################################
        # Normalized cumsum
        # (beware of the pytorch bug! Thus .clone()!)
        # ############################################
        input_cumsum: torch.Tensor = torch.cumsum(input, dim=1, dtype=input.dtype)
        input_cumsum_last: torch.Tensor = input_cumsum[:, -1, :, :].unsqueeze(1).clone()
        input_cumsum /= input_cumsum_last

        # ############################################
        # Get the required random numbers
        # ############################################
        random_values = torch.rand(
            size=[
                input_cumsum.shape[0],
                number_of_spikes,
                input_cumsum.shape[2],
                input_cumsum.shape[3],
            ],
            dtype=input.dtype,
            device=input.device,
        )

        # ############################################
        # Make space for the results
        # ############################################
        spikes = torch.empty_like(random_values, dtype=torch.int64, device=input.device)

        assert input_cumsum.is_contiguous() is True
        assert random_values.is_contiguous() is True
        assert spikes.is_contiguous() is True

        # time_start: float = time.perf_counter()
        spike_generation_profile = global_spike_generation_gpu_setting[
            spike_generation_gpu_setting_position
        ].clone()

        spike_generation_size = global_spike_size[
            spike_generation_gpu_setting_position
        ].clone()

        if input.device != torch.device("cpu"):
            if (
                (spike_generation_profile.numel() == 1)
                or (spike_generation_size[0] != int(spikes.shape[0]))
                or (spike_generation_size[1] != int(spikes.shape[1]))
                or (spike_generation_size[2] != int(spikes.shape[2]))
                or (spike_generation_size[3] != int(spikes.shape[3]))
            ):
                spike_generation_profile = torch.zeros(
                    (1, 7), dtype=torch.int64, device=torch.device("cpu")
                )

                global_spike_generation_cpp[
                    spike_generation_cpp_position
                ].gpu_occupancy_export(
                    int(spikes.shape[2]),
                    int(spikes.shape[3]),
                    int(spikes.shape[0]),
                    int(spikes.shape[1]),
                    spike_generation_profile.data_ptr(),
                    int(spike_generation_profile.shape[0]),
                    int(spike_generation_profile.shape[1]),
                )
                global_spike_generation_gpu_setting[
                    spike_generation_gpu_setting_position
                ] = spike_generation_profile.clone()

                spike_generation_size[0] = int(spikes.shape[0])
                spike_generation_size[1] = int(spikes.shape[1])
                spike_generation_size[2] = int(spikes.shape[2])
                spike_generation_size[3] = int(spikes.shape[3])
                global_spike_size[
                    spike_generation_gpu_setting_position
                ] = spike_generation_size.clone()

            else:
                global_spike_generation_cpp[
                    spike_generation_cpp_position
                ].gpu_occupancy_import(
                    spike_generation_profile.data_ptr(),
                    int(spike_generation_profile.shape[0]),
                    int(spike_generation_profile.shape[1]),
                )

        global_spike_generation_cpp[spike_generation_cpp_position].spike_generation(
            input_cumsum.data_ptr(),
            int(input_cumsum.shape[0]),
            int(input_cumsum.shape[1]),
            int(input_cumsum.shape[2]),
            int(input_cumsum.shape[3]),
            random_values.data_ptr(),
            int(random_values.shape[0]),
            int(random_values.shape[1]),
            int(random_values.shape[2]),
            int(random_values.shape[3]),
            spikes.data_ptr(),
            int(spikes.shape[0]),
            int(spikes.shape[1]),
            int(spikes.shape[2]),
            int(spikes.shape[3]),
            int(spike_number_of_cpu_processes),
        )
        del random_values
        del input_cumsum

        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        grad_parameter_list = None
        return (grad_input, grad_parameter_list)


class FunctionalSbS(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        spikes: torch.Tensor,
        epsilon_xy: torch.Tensor,
        epsilon_t_0: torch.Tensor,
        weights: torch.Tensor,
        h_initial: torch.Tensor,
        parameter_list: torch.Tensor,
        grad_output_scale: torch.Tensor,
        forgetting_offset: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        number_of_spikes: int = int(parameter_list[4])

        if input.device == torch.device("cpu"):
            hdyn_number_of_cpu_processes: int = int(parameter_list[5])
        else:
            hdyn_number_of_cpu_processes = -1

        output_size_0: int = int(parameter_list[6])
        output_size_1: int = int(parameter_list[7])
        gpu_tuning_factor: int = int(parameter_list[8])

        sbs_gpu_setting_position = int(parameter_list[11])
        sbs_hdynamic_cpp_position = int(parameter_list[12])

        # ###########################################################
        # H dynamic
        # ###########################################################

        assert epsilon_t_0.ndim == 1
        assert epsilon_t_0.shape[0] >= number_of_spikes

        # ############################################
        # Make space for the results
        # ############################################

        output = torch.empty(
            (
                int(input.shape[0]),
                int(weights.shape[1]),
                output_size_0,
                output_size_1,
            ),
            dtype=input.dtype,
            device=input.device,
        )

        assert output.is_contiguous() is True
        assert epsilon_xy.is_contiguous() is True
        assert epsilon_t_0.is_contiguous() is True
        assert weights.is_contiguous() is True
        assert spikes.is_contiguous() is True
        assert h_initial.is_contiguous() is True

        assert epsilon_xy.ndim == 3
        assert weights.ndim == 2
        assert h_initial.ndim == 1

        sbs_profile = global_sbs_gpu_setting[sbs_gpu_setting_position].clone()

        sbs_size = global_sbs_size[sbs_gpu_setting_position].clone()

        if input.device != torch.device("cpu"):
            if (
                (sbs_profile.numel() == 1)
                or (sbs_size[0] != int(output.shape[0]))
                or (sbs_size[1] != int(output.shape[1]))
                or (sbs_size[2] != int(output.shape[2]))
                or (sbs_size[3] != int(output.shape[3]))
            ):
                sbs_profile = torch.zeros(
                    (14, 7), dtype=torch.int64, device=torch.device("cpu")
                )

                global_sbs_hdynamic_cpp[sbs_hdynamic_cpp_position].gpu_occupancy_export(
                    int(output.shape[2]),
                    int(output.shape[3]),
                    int(output.shape[0]),
                    int(output.shape[1]),
                    sbs_profile.data_ptr(),
                    int(sbs_profile.shape[0]),
                    int(sbs_profile.shape[1]),
                )
                global_sbs_gpu_setting[sbs_gpu_setting_position] = sbs_profile.clone()
                sbs_size[0] = int(output.shape[0])
                sbs_size[1] = int(output.shape[1])
                sbs_size[2] = int(output.shape[2])
                sbs_size[3] = int(output.shape[3])
                global_sbs_size[sbs_gpu_setting_position] = sbs_size.clone()

            else:
                global_sbs_hdynamic_cpp[sbs_hdynamic_cpp_position].gpu_occupancy_import(
                    sbs_profile.data_ptr(),
                    int(sbs_profile.shape[0]),
                    int(sbs_profile.shape[1]),
                )

        global_sbs_hdynamic_cpp[sbs_hdynamic_cpp_position].update(
            output.data_ptr(),
            int(output.shape[0]),
            int(output.shape[1]),
            int(output.shape[2]),
            int(output.shape[3]),
            epsilon_xy.data_ptr(),
            int(epsilon_xy.shape[0]),
            int(epsilon_xy.shape[1]),
            int(epsilon_xy.shape[2]),
            epsilon_t_0.data_ptr(),
            int(epsilon_t_0.shape[0]),
            weights.data_ptr(),
            int(weights.shape[0]),
            int(weights.shape[1]),
            spikes.data_ptr(),
            int(spikes.shape[0]),
            int(spikes.shape[1]),
            int(spikes.shape[2]),
            int(spikes.shape[3]),
            h_initial.data_ptr(),
            int(h_initial.shape[0]),
            hdyn_number_of_cpu_processes,
            float(forgetting_offset.item()),
            int(gpu_tuning_factor),
        )

        # ###########################################################
        # Save the necessary data for the backward pass
        # ###########################################################

        ctx.save_for_backward(
            input,
            weights,
            output,
            parameter_list,
            grad_output_scale,
            labels,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # ##############################################
        # Get the variables back
        # ##############################################
        (
            input,
            weights,
            output,
            parameter_list,
            last_grad_scale,
            labels,
        ) = ctx.saved_tensors

        assert labels.numel() > 0

        # ##############################################
        # Default output
        # ##############################################
        grad_input = None
        grad_spikes = None
        grad_eps_xy = None
        grad_epsilon_t_0 = None
        grad_weights = None
        grad_h_initial = None
        grad_parameter_list = None
        grad_forgetting_offset = None
        grad_labels = None

        # ##############################################
        # Parameters
        # ##############################################
        parameter_w_trainable: bool = bool(parameter_list[0])
        parameter_disable_scale_grade: bool = bool(parameter_list[1])
        parameter_keep_last_grad_scale: bool = bool(parameter_list[2])
        parameter_skip_gradient_calculation: bool = bool(parameter_list[3])
        parameter_output_layer: bool = bool(parameter_list[9])
        parameter_local_learning: bool = bool(parameter_list[10])

        # ##############################################
        # Dealing with overall scale of the gradient
        # ##############################################
        if parameter_disable_scale_grade is False:
            if parameter_keep_last_grad_scale is True:
                last_grad_scale = torch.tensor(
                    [torch.abs(grad_output).max(), last_grad_scale]
                ).max()
            grad_output /= last_grad_scale
        grad_output_scale = last_grad_scale.clone()

        input /= input.sum(dim=1, keepdim=True, dtype=weights.dtype)

        # #################################################
        # User doesn't want us to calculate the gradients
        # #################################################

        if parameter_skip_gradient_calculation is True:

            return (
                grad_input,
                grad_spikes,
                grad_eps_xy,
                grad_epsilon_t_0,
                grad_weights,
                grad_h_initial,
                grad_parameter_list,
                grad_output_scale,
                grad_forgetting_offset,
                grad_labels,
            )

        # #################################################
        # Calculate backprop error (grad_input)
        # #################################################

        backprop_r: torch.Tensor = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1
        ) * output.unsqueeze(1)

        backprop_bigr: torch.Tensor = backprop_r.sum(dim=2)

        backprop_z: torch.Tensor = backprop_r * (
            1.0 / (backprop_bigr + 1e-20)
        ).unsqueeze(2)
        grad_input: torch.Tensor = (backprop_z * grad_output.unsqueeze(1)).sum(2)
        del backprop_z

        # #################################################
        # Calculate weight gradient (grad_weights)
        # #################################################

        if parameter_w_trainable is False:

            # #################################################
            # We don't train this weight
            # #################################################
            grad_weights = None

        elif (parameter_output_layer is False) and (parameter_local_learning is True):
            # #################################################
            # Local learning
            # #################################################
            grad_weights = (
                (-2 * (input - backprop_bigr).unsqueeze(2) * output.unsqueeze(1))
                .sum(0)
                .sum(-1)
                .sum(-1)
            )

        elif (parameter_output_layer is True) and (parameter_local_learning is True):

            target_one_hot: torch.Tensor = torch.zeros(
                (
                    labels.shape[0],
                    output.shape[1],
                ),
                device=input.device,
                dtype=input.dtype,
            )

            target_one_hot.scatter_(
                1,
                labels.to(input.device).unsqueeze(1),
                torch.ones(
                    (labels.shape[0], 1),
                    device=input.device,
                    dtype=input.dtype,
                ),
            )
            target_one_hot = target_one_hot.unsqueeze(-1).unsqueeze(-1)

            # (-2 * (input - backprop_bigr).unsqueeze(2) * (target_one_hot-output).unsqueeze(1))
            # (-2 * input.unsqueeze(2) * (target_one_hot-output).unsqueeze(1))
            grad_weights = (
                (
                    -2
                    * (input - backprop_bigr).unsqueeze(2)
                    * target_one_hot.unsqueeze(1)
                )
                .sum(0)
                .sum(-1)
                .sum(-1)
            )

        else:
            # #################################################
            # Backprop
            # #################################################
            backprop_f: torch.Tensor = output.unsqueeze(1) * (
                input / (backprop_bigr**2 + 1e-20)
            ).unsqueeze(2)

            result_omega: torch.Tensor = backprop_bigr.unsqueeze(
                2
            ) * grad_output.unsqueeze(1)
            result_omega -= (backprop_r * grad_output.unsqueeze(1)).sum(2).unsqueeze(2)
            result_omega *= backprop_f
            del backprop_f
            grad_weights = result_omega.sum(0).sum(-1).sum(-1)
            del result_omega

        del backprop_bigr
        del backprop_r

        return (
            grad_input,
            grad_spikes,
            grad_eps_xy,
            grad_epsilon_t_0,
            grad_weights,
            grad_h_initial,
            grad_parameter_list,
            grad_output_scale,
            grad_forgetting_offset,
            grad_labels,
        )
