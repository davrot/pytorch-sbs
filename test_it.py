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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import sys
import torch
import time
import dataconf
import logging
from datetime import datetime
import glob

from Dataset import (
    DatasetMaster,
    DatasetCIFAR,
    DatasetMNIST,
    DatasetFashionMNIST,
)
from Parameter import Config
from SbS import SbS


#######################################################################
# We want to log what is going on into a file and screen              #
#######################################################################

now = datetime.now()
dt_string_filename = now.strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(
    filename="log_" + dt_string_filename + ".txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

#######################################################################
# Load the config data from the json file                             #
#######################################################################

if len(sys.argv) < 2:
    raise Exception("Argument: Config file name is missing")

if len(sys.argv) < 3:
    raise Exception("Argument: Weight and epsilon file id is missing")


filename: str = sys.argv[1]

if os.path.exists(filename) is False:
    raise Exception(f"Config file not found! {filename}")

cfg = dataconf.file(filename, Config)
logging.info(f"Using configuration file: {filename}")

cfg.learning_step = int(sys.argv[2])
assert cfg.learning_step > 0

#######################################################################
# Prepare the test and training data                                  #
#######################################################################

# Load the input data
the_dataset_test: DatasetMaster
if cfg.data_mode == "CIFAR10":
    the_dataset_test = DatasetCIFAR(
        train=False, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
elif cfg.data_mode == "MNIST":
    the_dataset_test = DatasetMNIST(
        train=False, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
elif cfg.data_mode == "MNIST_FASHION":
    the_dataset_test = DatasetFashionMNIST(
        train=False, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
else:
    raise Exception("data_mode unknown")

cfg.image_statistics.mean = the_dataset_test.mean

# The basic size
cfg.image_statistics.the_size = [
    the_dataset_test.pattern_storage.shape[2],
    the_dataset_test.pattern_storage.shape[3],
]

# Minus the stuff we cut away in the pattern filter
cfg.image_statistics.the_size[0] -= 2 * cfg.augmentation.crop_width_in_pixel
cfg.image_statistics.the_size[1] -= 2 * cfg.augmentation.crop_width_in_pixel

my_loader_test: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    the_dataset_test, batch_size=cfg.batch_size, shuffle=False
)

logging.info("*** Data loaded.")

#######################################################################
# Build the network                                                   #
#######################################################################

wf: list[np.ndarray] = []
eps_xy: list[np.ndarray] = []
network = torch.nn.Sequential()
for id in range(0, len(cfg.network_structure.is_pooling_layer)):
    if id == 0:
        input_size: list[int] = cfg.image_statistics.the_size
    else:
        input_size = network[id - 1].output_size.tolist()

    network.append(
        SbS(
            number_of_input_neurons=cfg.network_structure.forward_neuron_numbers[id][0],
            number_of_neurons=cfg.network_structure.forward_neuron_numbers[id][1],
            input_size=input_size,
            forward_kernel_size=cfg.network_structure.forward_kernel_size[id],
            number_of_spikes=cfg.number_of_spikes,
            epsilon_t=cfg.get_epsilon_t(),
            epsilon_xy_intitial=cfg.learning_parameters.eps_xy_intitial,
            epsilon_0=cfg.epsilon_0,
            weight_noise_amplitude=cfg.learning_parameters.weight_noise_amplitude,
            is_pooling_layer=cfg.network_structure.is_pooling_layer[id],
            strides=cfg.network_structure.strides[id],
            dilation=cfg.network_structure.dilation[id],
            padding=cfg.network_structure.padding[id],
            alpha_number_of_iterations=cfg.learning_parameters.alpha_number_of_iterations,
            number_of_cpu_processes=cfg.number_of_cpu_processes,
        )
    )

    eps_xy.append(network[id].epsilon_xy.detach().clone().numpy())
    wf.append(network[id].weights.detach().clone().numpy())

logging.info("*** Network generated.")


for id in range(0, len(network)):

    # Are there weights that overwrite the initial weights?
    file_to_load = glob.glob(
        cfg.learning_parameters.overload_path + "/Weight_L" + str(id) + "*.npy"
    )

    if len(file_to_load) > 1:
        raise Exception(
            f"Too many previous weights files {cfg.learning_parameters.overload_path}/Weight_L{id}*.npy"
        )

    if len(file_to_load) == 1:
        network[id].weights = torch.tensor(
            np.load(file_to_load[0]),
            dtype=torch.float32,
        )
        wf[id] = np.load(file_to_load[0])
        logging.info(f"File used: {file_to_load[0]}")

    # Are there epsinlon xy files that overwrite the initial epsilon xy?
    file_to_load = glob.glob(
        cfg.learning_parameters.overload_path + "/EpsXY_L" + str(id) + "*.npy"
    )

    if len(file_to_load) > 1:
        raise Exception(
            f"Too many previous epsilon xy files {cfg.learning_parameters.overload_path}/EpsXY_L{id}*.npy"
        )

    if len(file_to_load) == 1:
        network[id].epsilon_xy = torch.tensor(
            np.load(file_to_load[0]),
            dtype=torch.float32,
        )
        eps_xy[id] = np.load(file_to_load[0])
        logging.info(f"File used: {file_to_load[0]}")

for id in range(0, len(network)):
    # Load previous weights and epsilon xy
    if cfg.learning_step > 0:
        filename = (
            cfg.weight_path
            + "/Weight_L"
            + str(id)
            + "_S"
            + str(cfg.learning_step)
            + ".npy"
        )
        if os.path.exists(filename) is True:
            network[id].weights = torch.tensor(
                np.load(filename),
                dtype=torch.float32,
            )
            wf[id] = np.load(filename)

        filename = (
            cfg.eps_xy_path
            + "/EpsXY_L"
            + str(id)
            + "_S"
            + str(cfg.learning_step)
            + ".npy"
        )
        if os.path.exists(filename) is True:
            network[id].epsilon_xy = torch.tensor(
                np.load(filename),
                dtype=torch.float32,
            )
            eps_xy[id] = np.load(filename)


#######################################################################
# Some variable declarations                                          #
#######################################################################

test_correct: int = 0
test_all: int = 0
test_complete: int = the_dataset_test.__len__()

logging.info("")

with torch.no_grad():
    logging.info("Testing:")

    for h_x, h_x_labels in my_loader_test:
        time_0 = time.perf_counter()

        h_h: torch.Tensor = network(
            the_dataset_test.pattern_filter_test(h_x, cfg).type(dtype=torch.float32)
        )

        test_correct += (h_h.argmax(dim=1).squeeze() == h_x_labels).sum().numpy()
        test_all += h_h.shape[0]
        performance = 100.0 * test_correct / test_all
        time_1 = time.perf_counter()
        time_measure_a = time_1 - time_0

        logging.info(
            (
                f"\t\t{test_all} of {test_complete}"
                f" with {performance/100:^6.2%} \t Time used: {time_measure_a:^6.2f}sec"
            )
        )
        np_performance = np.array(performance)
        np.save(f"{cfg.results_path}/{cfg.learning_step}.npy", np_performance)


# %%
