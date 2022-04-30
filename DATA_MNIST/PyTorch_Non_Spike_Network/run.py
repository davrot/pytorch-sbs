# %%
import torch
from Dataset import DatasetMNIST

# Some parameters
input_number_of_channel: int = 1
input_dim_x: int = 24
input_dim_y: int = 24

number_of_output_channels_conv1: int = 32
number_of_output_channels_conv2: int = 64
number_of_output_channels_flatten1: int = 576
number_of_output_channels_full1: int = 10

kernel_size_conv1: tuple[int, int] = (5, 5)
kernel_size_pool1: tuple[int, int] = (2, 2)
kernel_size_conv2: tuple[int, int] = (5, 5)
kernel_size_pool2: tuple[int, int] = (2, 2)

stride_conv1: tuple[int, int] = (1, 1)
stride_pool1: tuple[int, int] = (2, 2)
stride_conv2: tuple[int, int] = (1, 1)
stride_pool2: tuple[int, int] = (2, 2)

padding_conv1: int = 0
padding_pool1: int = 0
padding_conv2: int = 0
padding_pool2: int = 0

network = torch.nn.Sequential(
    torch.nn.Conv2d(
        in_channels=input_number_of_channel,
        out_channels=number_of_output_channels_conv1,
        kernel_size=kernel_size_conv1,
        stride=stride_conv1,
        padding=padding_conv1,
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(
        kernel_size=kernel_size_pool1, stride=stride_pool1, padding=padding_pool1
    ),
    torch.nn.Conv2d(
        in_channels=number_of_output_channels_conv1,
        out_channels=number_of_output_channels_conv2,
        kernel_size=kernel_size_conv2,
        stride=stride_conv2,
        padding=padding_conv2,
    ),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(
        kernel_size=kernel_size_pool2, stride=stride_pool2, padding=padding_pool2
    ),
    torch.nn.Flatten(
        start_dim=1,
    ),
    torch.nn.Linear(
        in_features=number_of_output_channels_flatten1,
        out_features=number_of_output_channels_full1,
        bias=True,
    ),
    torch.nn.Softmax(dim=1),
)
# %%
import torchvision

test_processing_chain = torchvision.transforms.Compose(
    transforms=[torchvision.transforms.CenterCrop((24, 24))],
)
train_processing_chain = torchvision.transforms.Compose(
    transforms=[torchvision.transforms.RandomCrop((24, 24))],
)

path_pattern: str = "./DATA_MNIST/"
path_label: str = "./DATA_MNIST/"

dataset_train = DatasetMNIST(
    train=True, path_pattern=path_pattern, path_label=path_label
)
dataset_test = DatasetMNIST(
    train=False, path_pattern=path_pattern, path_label=path_label
)
batch_size_train: int = 100
batch_size_test: int = 100


train_data_load = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size_train, shuffle=True
)

test_data_load = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size_test, shuffle=False
)

# %%
# The optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
# The LR Scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75)

# %%
number_of_test_pattern: int = dataset_test.__len__()
number_of_train_pattern: int = dataset_train.__len__()

number_of_epoch: int = 50

# %%
import time
from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()

# %%
loss_function = torch.nn.CrossEntropyLoss()

for epoch_id in range(0, number_of_epoch):
    print(f"Epoch: {epoch_id}")
    t_start: float = time.perf_counter()

    train_loss: float = 0.0
    train_correct: int = 0
    train_number: int = 0
    test_correct: int = 0
    test_number: int = 0

    # Switch the network into training mode
    network.train()

    # This runs in total for one epoch split up into mini-batches
    for image, target in train_data_load:

        # Clean the gradient
        optimizer.zero_grad()

        output = network(train_processing_chain(image))

        loss = loss_function(output, target)

        train_loss += loss.item()
        train_correct += (output.argmax(dim=1) == target).sum().numpy()
        train_number += target.shape[0]
        # Calculate backprop
        loss.backward()

        # Update the parameter
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step(train_loss)

    t_training: float = time.perf_counter()

    # Switch the network into evalution mode
    network.eval()
    with torch.no_grad():
        for image, target in test_data_load:

            output = network(test_processing_chain(image))

            test_correct += (output.argmax(dim=1) == target).sum().numpy()
            test_number += target.shape[0]

    t_testing = time.perf_counter()

    perfomance_test_correct: float = 100.0 * test_correct / test_number
    perfomance_train_correct: float = 100.0 * train_correct / train_number

    tb.add_scalar("Train Loss", train_loss, epoch_id)
    tb.add_scalar("Train Number Correct", train_correct, epoch_id)
    tb.add_scalar("Test Number Correct", test_correct, epoch_id)

    print(f"Training: Loss={train_loss:.5f} Correct={perfomance_train_correct:.2f}%")
    print(f"Testing: Correct={perfomance_test_correct:.2f}%")
    print(
        f"Time: Training={(t_training-t_start):.1f}sec, Testing={(t_testing-t_training):.1f}sec"
    )
    torch.save(network, "Model_MNIST_A_" + str(epoch_id) + ".pt")
    print()

# %%
tb.close()
