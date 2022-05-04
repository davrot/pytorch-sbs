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

from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()

torch.set_default_dtype(torch.float32)

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

filename: str = sys.argv[1]

if os.path.exists(filename) is False:
    raise Exception(f"Config file not found! {filename}")

cfg = dataconf.file(filename, Config)
logging.info(f"Using configuration file: {filename}")


#######################################################################
# Prepare the test and training data                                  #
#######################################################################

# Load the input data
the_dataset_train: DatasetMaster
the_dataset_test: DatasetMaster
if cfg.data_mode == "CIFAR10":
    the_dataset_train = DatasetCIFAR(
        train=True, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
    the_dataset_test = DatasetCIFAR(
        train=False, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
elif cfg.data_mode == "MNIST":
    the_dataset_train = DatasetMNIST(
        train=True, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
    the_dataset_test = DatasetMNIST(
        train=False, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
elif cfg.data_mode == "MNIST_FASHION":
    the_dataset_train = DatasetFashionMNIST(
        train=True, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
    the_dataset_test = DatasetFashionMNIST(
        train=False, path_pattern=cfg.data_path, path_label=cfg.data_path
    )
else:
    raise Exception("data_mode unknown")

if len(cfg.image_statistics.mean) == 0:
    cfg.image_statistics.mean = the_dataset_train.mean

# The basic size
cfg.image_statistics.the_size = [
    the_dataset_train.pattern_storage.shape[2],
    the_dataset_train.pattern_storage.shape[3],
]

# Minus the stuff we cut away in the pattern filter
cfg.image_statistics.the_size[0] -= 2 * cfg.augmentation.crop_width_in_pixel
cfg.image_statistics.the_size[1] -= 2 * cfg.augmentation.crop_width_in_pixel

my_loader_test: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    the_dataset_test, batch_size=cfg.batch_size, shuffle=False
)
my_loader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    the_dataset_train, batch_size=cfg.batch_size, shuffle=True
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

#######################################################################
# Optimizer and LR Scheduler                                          #
#######################################################################

# I keep weights and epsilon xy seperate to
# set the initial learning rate independently
parameter_list_weights: list = []
parameter_list_epsilon_xy: list = []

for id in range(0, len(network)):
    parameter_list_weights.append(network[id]._weights)
    parameter_list_epsilon_xy.append(network[id]._epsilon_xy)

if cfg.learning_parameters.optimizer_name == "Adam":
    if cfg.learning_parameters.learning_rate_gamma_w > 0:
        optimizer_wf: torch.optim.Optimizer = torch.optim.Adam(
            parameter_list_weights,
            lr=cfg.learning_parameters.learning_rate_gamma_w,
        )
    else:
        optimizer_wf = torch.optim.Adam(
            parameter_list_weights,
        )

    if cfg.learning_parameters.learning_rate_gamma_eps_xy > 0:
        optimizer_eps: torch.optim.Optimizer = torch.optim.Adam(
            parameter_list_epsilon_xy,
            lr=cfg.learning_parameters.learning_rate_gamma_eps_xy,
        )
    else:
        optimizer_eps = torch.optim.Adam(
            parameter_list_epsilon_xy,
        )
else:
    raise Exception("Optimizer not implemented")

if cfg.learning_parameters.lr_schedule_name == "ReduceLROnPlateau":
    if cfg.learning_parameters.lr_scheduler_patience_w > 0:
        lr_scheduler_wf = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_wf,
            factor=cfg.learning_parameters.lr_scheduler_factor_w,
            patience=cfg.learning_parameters.lr_scheduler_patience_w,
        )

    if cfg.learning_parameters.lr_scheduler_patience_eps_xy > 0:
        lr_scheduler_eps = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_eps,
            factor=cfg.learning_parameters.lr_scheduler_factor_eps_xy,
            patience=cfg.learning_parameters.lr_scheduler_patience_eps_xy,
        )
else:
    raise Exception("lr_scheduler not implemented")

logging.info("*** Optimizer prepared.")


#######################################################################
# Some variable declarations                                          #
#######################################################################

test_correct: int = 0
test_all: int = 0
test_complete: int = the_dataset_test.__len__()

train_correct: int = 0
train_all: int = 0
train_complete: int = the_dataset_train.__len__()

train_number_of_processed_pattern: int = 0

train_loss: np.ndarray = np.zeros((1), dtype=np.float32)

last_test_performance: float = -1.0


logging.info("")

with torch.no_grad():
    if cfg.learning_parameters.learning_active is True:
        while True:

            ###############################################
            # Run a training data batch                   #
            ###############################################

            for h_x, h_x_labels in my_loader_train:
                time_0: float = time.perf_counter()

                if train_number_of_processed_pattern == 0:
                    # Reset the gradient of the torch optimizers
                    optimizer_wf.zero_grad()
                    optimizer_eps.zero_grad()

                with torch.enable_grad():

                    h_collection = []
                    h_collection.append(
                        the_dataset_train.pattern_filter_train(h_x, cfg).type(
                            dtype=torch.float32
                        )
                    )
                    for id in range(0, len(network)):
                        h_collection.append(network[id](h_collection[-1]))

                    # Convert label into one hot
                    target_one_hot: torch.Tensor = torch.zeros(
                        (
                            h_x_labels.shape[0],
                            int(cfg.network_structure.number_of_output_neurons),
                        )
                    )
                    target_one_hot.scatter_(
                        1, h_x_labels.unsqueeze(1), torch.ones((h_x_labels.shape[0], 1))
                    )
                    target_one_hot = (
                        target_one_hot.unsqueeze(2)
                        .unsqueeze(2)
                        .type(dtype=torch.float32)
                    )

                    h_y1 = torch.log(h_collection[-1] + 1e-20)

                    my_loss: torch.Tensor = (
                        (
                            torch.nn.functional.mse_loss(
                                h_collection[-1],
                                target_one_hot,
                                reduction="none",
                            )
                            * cfg.learning_parameters.loss_coeffs_mse
                            + torch.nn.functional.kl_div(
                                h_y1, target_one_hot + 1e-20, reduction="none"
                            )
                            * cfg.learning_parameters.loss_coeffs_kldiv
                        )
                        / (
                            cfg.learning_parameters.loss_coeffs_kldiv
                            + cfg.learning_parameters.loss_coeffs_mse
                        )
                    ).mean()

                    time_1: float = time.perf_counter()

                    my_loss.backward()

                    my_loss_float = my_loss.item()
                    time_2: float = time.perf_counter()

                train_correct += (
                    (h_collection[-1].argmax(dim=1).squeeze() == h_x_labels)
                    .sum()
                    .numpy()
                )
                train_all += h_collection[-1].shape[0]

                performance: float = 100.0 * train_correct / train_all

                time_measure_a: float = time_1 - time_0

                logging.info(
                    (
                        f"{cfg.learning_step:^6} Training \t{train_all^6} pattern "
                        f"with {performance/100.0:^6.2%} "
                        f"\t\tForward time: \t{time_measure_a:^6.2f}sec"
                    )
                )

                train_loss[0] += my_loss_float
                train_number_of_processed_pattern += h_collection[-1].shape[0]

                time_measure_b: float = time_2 - time_1

                logging.info(
                    (
                        f"\t\t\tLoss: {train_loss[0]/train_number_of_processed_pattern:^15.3e} "
                        f"\t\t\tBackward time: \t{time_measure_b:^6.2f}sec "
                    )
                )

                if (
                    train_number_of_processed_pattern
                    >= cfg.get_update_after_x_pattern()
                ):
                    logging.info("\t\t\t*** Updating the weights ***")
                    my_loss_for_batch: float = (
                        train_loss[0] / train_number_of_processed_pattern
                    )

                    optimizer_wf.step()
                    optimizer_eps.step()

                    for id in range(0, len(network)):
                        if cfg.network_structure.w_trainable[id] is True:
                            network[id].norm_weights()
                            network[id].threshold_weights(
                                cfg.learning_parameters.learning_rate_threshold_w
                            )
                            network[id].norm_weights()
                        else:
                            network[id].weights = torch.tensor(
                                wf[id], dtype=torch.float32
                            )

                        if cfg.network_structure.eps_xy_trainable[id] is True:
                            network[id].threshold_epsilon_xy(
                                cfg.learning_parameters.learning_rate_threshold_eps_xy
                            )
                            if cfg.network_structure.eps_xy_mean[id] is True:
                                network[id].mean_epsilon_xy()
                        else:
                            network[id].epsilon_xy = torch.tensor(
                                eps_xy[id], dtype=torch.float32
                            )

                        if cfg.network_structure.w_trainable[id] is True:
                            # Save the new values
                            np.save(
                                cfg.weight_path
                                + "/Weight_L"
                                + str(id)
                                + "_S"
                                + str(cfg.learning_step)
                                + ".npy",
                                network[id].weights.detach().numpy(),
                            )

                            try:
                                tb.add_histogram(
                                    "Weights " + str(id),
                                    network[id].weights,
                                    cfg.learning_step,
                                )
                            except ValueError:
                                pass

                        if cfg.network_structure.eps_xy_trainable[id] is True:
                            np.save(
                                cfg.eps_xy_path
                                + "/EpsXY_L"
                                + str(id)
                                + "_S"
                                + str(cfg.learning_step)
                                + ".npy",
                                network[id].epsilon_xy.detach().numpy(),
                            )
                            try:
                                tb.add_histogram(
                                    "Epsilon XY " + str(id),
                                    network[id].epsilon_xy.detach().numpy(),
                                    cfg.learning_step,
                                )
                            except ValueError:
                                pass

                    # Let the torch learning rate scheduler update the
                    # learning rates of the optimiers
                    if cfg.learning_parameters.lr_scheduler_patience_w > 0:
                        if cfg.learning_parameters.lr_scheduler_use_performance is True:
                            lr_scheduler_wf.step(100.0 - performance)
                        else:
                            lr_scheduler_wf.step(my_loss_for_batch)

                    if cfg.learning_parameters.lr_scheduler_patience_eps_xy > 0:
                        if cfg.learning_parameters.lr_scheduler_use_performance is True:
                            lr_scheduler_eps.step(100.0 - performance)
                        else:
                            lr_scheduler_eps.step(my_loss_for_batch)

                    tb.add_scalar("Train Error", 100.0 - performance, cfg.learning_step)
                    tb.add_scalar("Train Loss", my_loss_for_batch, cfg.learning_step)
                    tb.add_scalar(
                        "Learning Rate Scale WF",
                        optimizer_wf.param_groups[-1]["lr"],
                        cfg.learning_step,
                    )
                    tb.add_scalar(
                        "Learning Rate Scale Eps XY ",
                        optimizer_eps.param_groups[-1]["lr"],
                        cfg.learning_step,
                    )

                    cfg.learning_step += 1
                    train_loss = np.zeros((1), dtype=np.float32)
                    train_correct = 0
                    train_all = 0
                    performance = 0
                    train_number_of_processed_pattern = 0

                    tb.flush()

                    test_correct = 0
                    test_all = 0

                    if last_test_performance < 0:
                        logging.info("")
                    else:
                        logging.info(
                            f"\t\t\tLast test performance: {last_test_performance/100.0:^6.2%}"
                        )
                    logging.info("")

                    ###############################################
                    # Run a test data performance measurement     #
                    ###############################################
                    if (
                        (
                            (
                                (
                                    cfg.learning_step
                                    % cfg.learning_parameters.test_every_x_learning_steps
                                )
                                == 0
                            )
                            or (cfg.learning_step == cfg.learning_step_max)
                        )
                        and (cfg.learning_parameters.test_during_learning is True)
                        and (cfg.learning_step > 0)
                    ):
                        logging.info("")
                        logging.info("Testing:")

                        for h_x, h_x_labels in my_loader_test:
                            time_0 = time.perf_counter()

                            h_h: torch.Tensor = network(
                                the_dataset_test.pattern_filter_test(h_x, cfg).type(
                                    dtype=torch.float32
                                )
                            )

                            test_correct += (
                                (h_h.argmax(dim=1).squeeze() == h_x_labels)
                                .sum()
                                .numpy()
                            )
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

                        logging.info("")

                        last_test_performance = performance

                        tb.add_scalar(
                            "Test Error", 100.0 - performance, cfg.learning_step
                        )
                        tb.flush()

                    if cfg.learning_step == cfg.learning_step_max:
                        tb.close()
                        exit(1)

# %%
