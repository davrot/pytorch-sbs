import torch


def calculate_output_size(
    value: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    padding: list[int],
) -> torch.Tensor:
    assert len(value) == 2
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(dilation) == 2
    assert len(padding) == 2

    coordinates_0, coordinates_1 = get_coordinates(
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
    )

    output_size: torch.Tensor = torch.tensor(
        [
            coordinates_0.shape[1],
            coordinates_1.shape[1],
        ],
        dtype=torch.int64,
    )
    return output_size


def get_coordinates(
    value: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    padding: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function converts parameter in coordinates
    for the convolution window"""

    coordinates_0: torch.Tensor = (
        torch.nn.functional.unfold(
            torch.arange(0, int(value[0]), dtype=torch.float32)
            .unsqueeze(1)
            .unsqueeze(0)
            .unsqueeze(0),
            kernel_size=(int(kernel_size[0]), 1),
            dilation=int(dilation[0]),
            padding=(int(padding[0]), 0),
            stride=int(stride[0]),
        )
        .squeeze(0)
        .type(torch.int64)
    )

    coordinates_1: torch.Tensor = (
        torch.nn.functional.unfold(
            torch.arange(0, int(value[1]), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0),
            kernel_size=(1, int(kernel_size[1])),
            dilation=int(dilation[1]),
            padding=(0, int(padding[1])),
            stride=int(stride[1]),
        )
        .squeeze(0)
        .type(torch.int64)
    )

    return coordinates_0, coordinates_1


if __name__ == "__main__":
    a, b = get_coordinates(
        value=[28, 28],
        kernel_size=[5, 5],
        stride=[1, 1],
        dilation=[1, 1],
        padding=[0, 0],
    )
    print(a.shape)
    print(b.shape)
