import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import json
from jsmin import jsmin
import glob

# -------------------------------

filename:str = "def.json"
with open(filename) as json_file:
    minified = jsmin(json_file.read())
data = json.loads(minified)
number_of_spikes = data["number_of_spikes"]


# -------------------------------


path_runs: str = "./Log/*"  

temp = glob.glob(path_runs)
assert len(temp) == 1
path = temp[0]


acc = event_accumulator.EventAccumulator(path)
acc.Reload()

available_scalar = acc.Tags()["scalars"]
available_histograms = acc.Tags()["histograms"]

which_scalar = "Test Error"
te = acc.Scalars(which_scalar)

temp = []
for te_item in te:
    temp.append((te_item[1], te_item[2]))
temp = np.array(temp)

print(temp)
np.save(f"test_error.npy", temp)

