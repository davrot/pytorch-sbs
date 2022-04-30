import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

filename: str = "events.out.tfevents.1651325827.fedora.115860.0"

acc = event_accumulator.EventAccumulator(filename)
acc.Reload()

# What is available?
# available_scalar = acc.Tags()["scalars"]
# print("Available Scalars")
# print(available_scalar)

which_scalar: str = "Test Number Correct"
te = acc.Scalars(which_scalar)

temp: list = []
for te_item in te:
    temp.append((te_item[1], te_item[2]))
temp_np = np.array(temp)

plt.semilogy(temp_np[:, 0], (1.0 - (temp_np[:, 1] / 10000)) * 100)
plt.xlabel("Epochs")
plt.ylabel("Error [%]")
plt.savefig("Error.png")
plt.show()
