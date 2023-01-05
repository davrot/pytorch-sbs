# %%
import torch
import glob
import numpy as np

from network.SbS import SbS
from network.SplitOnOffLayer import SplitOnOffLayer
from network.Conv2dApproximation import Conv2dApproximation


def load_previous_weights(
    network: torch.nn.Sequential,
    overload_path: str,
    logging,
    device: torch.device,
    default_dtype: torch.dtype,
) -> None:

    for id in range(0, len(network)):

        # #################################################
        # SbS
        # #################################################

        if isinstance(network[id], SbS) is True:
            # Are there weights that overwrite the initial weights?
            file_to_load = glob.glob(overload_path + "/Weight_L" + str(id) + "_*.npy")

            if len(file_to_load) > 1:
                raise Exception(
                    f"Too many previous weights files {overload_path}/Weight_L{id}*.npy"
                )

            if len(file_to_load) == 1:
                network[id].weights = torch.tensor(
                    np.load(file_to_load[0]),
                    dtype=default_dtype,
                    device=device,
                )
                logging.info(f"Weights file used for layer {id} : {file_to_load[0]}")

        if isinstance(network[id], torch.nn.modules.conv.Conv2d) is True:

            # #################################################
            # Conv2d weights
            # #################################################

            # Are there weights that overwrite the initial weights?
            file_to_load = glob.glob(overload_path + "/Weight_L" + str(id) + "_*.npy")

            if len(file_to_load) > 1:
                raise Exception(
                    f"Too many previous weights files {overload_path}/Weight_L{id}*.npy"
                )

            if len(file_to_load) == 1:
                network[id]._parameters["weight"].data = torch.tensor(
                    np.load(file_to_load[0]),
                    dtype=default_dtype,
                    device=device,
                )
                logging.info(f"Weights file used for layer {id} : {file_to_load[0]}")

            # #################################################
            # Conv2d bias
            # #################################################

            # Are there biases that overwrite the initial weights?
            file_to_load = glob.glob(overload_path + "/Bias_L" + str(id) + "_*.npy")

            if len(file_to_load) > 1:
                raise Exception(
                    f"Too many previous weights files {overload_path}/Weight_L{id}*.npy"
                )

            if len(file_to_load) == 1:
                network[id]._parameters["bias"].data = torch.tensor(
                    np.load(file_to_load[0]),
                    dtype=default_dtype,
                    device=device,
                )
                logging.info(f"Bias file used for layer {id} : {file_to_load[0]}")

        if isinstance(network[id], Conv2dApproximation) is True:

            # #################################################
            # Approximate Conv2d weights
            # #################################################

            # Are there weights that overwrite the initial weights?
            file_to_load = glob.glob(overload_path + "/Weight_L" + str(id) + "_*.npy")

            if len(file_to_load) > 1:
                raise Exception(
                    f"Too many previous weights files {overload_path}/Weight_L{id}*.npy"
                )

            if len(file_to_load) == 1:
                network[id].weights.data = torch.tensor(
                    np.load(file_to_load[0]),
                    dtype=default_dtype,
                    device=device,
                )
                logging.info(f"Weights file used for layer {id} : {file_to_load[0]}")

            # #################################################
            # Approximate Conv2d bias
            # #################################################

            # Are there biases that overwrite the initial weights?
            file_to_load = glob.glob(overload_path + "/Bias_L" + str(id) + "_*.npy")

            if len(file_to_load) > 1:
                raise Exception(
                    f"Too many previous weights files {overload_path}/Weight_L{id}*.npy"
                )

            if len(file_to_load) == 1:
                network[id].bias.data = torch.tensor(
                    np.load(file_to_load[0]),
                    dtype=default_dtype,
                    device=device,
                )
                logging.info(f"Bias file used for layer {id} : {file_to_load[0]}")

        # #################################################
        # SplitOnOffLayer
        # #################################################
        if isinstance(network[id], SplitOnOffLayer) is True:
            # Are there weights that overwrite the initial weights?
            file_to_load = glob.glob(overload_path + "/Mean_L" + str(id) + "_*.npy")

            if len(file_to_load) > 1:
                raise Exception(
                    f"Too many previous mean files {overload_path}/Mean_L{id}*.npy"
                )

            if len(file_to_load) == 1:
                network[id].mean = torch.tensor(
                    np.load(file_to_load[0]),
                    dtype=default_dtype,
                    device=device,
                )
                logging.info(f"Meanfile used for layer {id} : {file_to_load[0]}")
