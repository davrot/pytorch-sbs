# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import torch
import dataconf
import logging
from datetime import datetime

from network.Parameter import Config

from network.build_network import build_network
from network.build_optimizer import build_optimizer
from network.build_lr_scheduler import build_lr_scheduler
from network.build_datasets import build_datasets
from network.load_previous_weights import load_previous_weights

from network.loop_train_test import (
    loop_test,
    loop_train,
    run_lr_scheduler,
    loop_test_reconstruction,
)
from network.SbSReconstruction import SbSReconstruction

from torch.utils.tensorboard import SummaryWriter


# ######################################################################
# We want to log what is going on into a file and screen
# ######################################################################

now = datetime.now()
dt_string_filename = now.strftime("%Y_%m_%d_%H_%M_%S")
logging.basicConfig(
    filename="log_" + dt_string_filename + ".txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

# ######################################################################
# Load the config data from the json file
# ######################################################################

if os.path.exists("def.json") is False:
    raise Exception("Config file not found! def.json")

if os.path.exists("network.json") is False:
    raise Exception("Config file not found! network.json")

if os.path.exists("dataset.json") is False:
    raise Exception("Config file not found! dataset.json")


cfg = dataconf.multi.file("network.json").file("dataset.json").file("def.json").on(Config)
logging.info(cfg)

logging.info(f"Number of spikes: {cfg.number_of_spikes}")
logging.info(f"Cooldown after spikes: {cfg.cooldown_after_number_of_spikes}")
logging.info(f"Reduction cooldown: {cfg.reduction_cooldown}")
logging.info("")
logging.info(f"Epsilon 0: {cfg.epsilon_0}")
logging.info(f"Batch size: {cfg.batch_size}")
logging.info(f"Data mode: {cfg.data_mode}")
logging.info("")
logging.info("*** Config loaded.")
logging.info("")

tb = SummaryWriter(log_dir=cfg.log_path)

# ###########################################
# GPU Yes / NO ?
# ###########################################
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
torch_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
use_gpu: bool = True if torch.cuda.is_available() else False
logging.info(f"Using {torch_device} device")
device = torch.device(torch_device)

# ######################################################################
# Prepare the test and training data
# ######################################################################

the_dataset_train, the_dataset_test, my_loader_test, my_loader_train = build_datasets(
    cfg
)

logging.info("*** Data loaded.")

# ######################################################################
# Build the network, Optimizer, and LR Scheduler                                                                   #
# ######################################################################

network = build_network(
    cfg=cfg, device=device, default_dtype=default_dtype, logging=logging
)
logging.info("")

optimizer = build_optimizer(network=network, cfg=cfg, logging=logging)

lr_scheduler = build_lr_scheduler(optimizer=optimizer, cfg=cfg, logging=logging)

logging.info("*** Network generated.")

load_previous_weights(
    network=network,
    overload_path=cfg.learning_parameters.overload_path,
    logging=logging,
    device=device,
    default_dtype=default_dtype,
)

logging.info("")

last_test_performance: float = -1.0


# ##############################################
# Run test data
# ##############################################
network.eval()
if isinstance(network[-1], SbSReconstruction) is False:
    last_test_performance = loop_test(
        epoch_id=cfg.epoch_id,
        cfg=cfg,
        network=network,
        my_loader_test=my_loader_test,
        the_dataset_test=the_dataset_test,
        device=device,
        default_dtype=default_dtype,
        logging=logging,
        tb=tb,
    )
else:
    last_test_performance = loop_test_reconstruction(
        epoch_id=cfg.epoch_id,
        cfg=cfg,
        network=network,
        my_loader_test=my_loader_test,
        the_dataset_test=the_dataset_test,
        device=device,
        default_dtype=default_dtype,
        logging=logging,
        tb=tb,
    )

tb.close()




# %%
