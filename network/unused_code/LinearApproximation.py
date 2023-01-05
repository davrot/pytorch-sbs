import torch
import math

from network.CPP.PyMultiApp import MultiApp


class LinearApproximation(torch.nn.Module):

    in_features: int | None = None
    out_features: int | None = None
    use_bias: bool = False

    approximation_enable: bool = False
    number_of_trunc_bits: int = -1
    number_of_frac: int = -1

    number_of_processes: int = 1

    weights: torch.nn.parameter.Parameter
    bias: torch.nn.parameter.Parameter | None

    device: torch.device
    dtype: torch.dtype

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        approximation_enable: bool = False,
        number_of_trunc_bits: int = -1,
        number_of_frac: int = -1,
        number_of_processes: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        assert device is not None
        self.device = device

        assert dtype is not None
        self.dtype = dtype

        self.in_features = in_features
        self.out_channels = out_features
        self.use_bias = bias

        self.approximation_enable = approximation_enable
        self.number_of_trunc_bits = number_of_trunc_bits
        self.number_of_frac = number_of_frac

        self.number_of_processes = number_of_processes

        if self.use_bias is True:
            self.bias: torch.nn.parameter.Parameter | None = (
                torch.nn.parameter.Parameter(
                    torch.empty(
                        (out_features),
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            )
        else:
            self.bias = None

        self.weights: torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=self.dtype,
                device=self.device,
            )
        )

        self.functional_multi = FunctionalMultiLinear.apply

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Stolen from original torch conv2 code
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        assert input.dim() == 2

        parameter_list = torch.tensor(
            [
                int(self.approximation_enable),  # 0
                int(self.number_of_trunc_bits),  # 1
                int(self.number_of_frac),  # 2
                int(self.number_of_processes),  # 3
            ],
            dtype=torch.int64,
        )

        output = self.functional_multi(
            input.unsqueeze(-1).unsqueeze(-1), self.weights, parameter_list
        )

        output = output.squeeze(-1).squeeze(-1)

        if self.bias is not None:
            output += self.bias.unsqueeze(0)

        return output


class FunctionalMultiLinear(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        parameter_list: torch.Tensor,
    ) -> torch.Tensor:

        assert input.ndim == 4
        assert input.dtype is torch.float32
        assert input.is_contiguous() is True

        assert weights.ndim == 2
        assert weights.dtype is torch.float32
        assert weights.is_contiguous() is True

        assert input.shape[1] == weights.shape[1]

        approximation_enable = bool(parameter_list[0])
        number_of_trunc_bits = int(parameter_list[1])
        number_of_frac = int(parameter_list[2])
        number_of_processes = int(parameter_list[3])

        assert input.device == weights.device

        output = torch.zeros(
            (input.shape[0], weights.shape[0], input.shape[2], input.shape[3]),
            dtype=weights.dtype,
            device=weights.device,
            requires_grad=True,
        )
        assert output.is_contiguous() is True

        multiplier: MultiApp = MultiApp()

        multiplier.update_with_init_vector_multi_pattern(
            input.data_ptr(),
            weights.data_ptr(),
            output.data_ptr(),
            int(output.shape[0]),  # pattern
            int(output.shape[1]),  # feature channel
            int(output.shape[2]),  # x
            int(output.shape[3]),  # y
            int(input.shape[1]),  # input channel
            int(number_of_processes),
            bool(approximation_enable),
            int(number_of_trunc_bits),
            int(number_of_frac),
        )

        ctx.save_for_backward(
            input.detach(),
            weights.detach(),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        (input, weights) = ctx.saved_tensors

        grad_input = (
            grad_output.unsqueeze(2) * weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        ).sum(1)
        grad_weights = (
            (grad_output.unsqueeze(2) * input.unsqueeze(1)).sum(0).sum(-1).sum(-1)
        )
        grad_parameter_list = None

        return (grad_input, grad_weights, grad_parameter_list)
