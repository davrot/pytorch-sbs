import numpy as np
import glob
import os
from natsort import natsorted


path: str = "noisy_picture_data"
spike_list: list[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
]

for spikes in spike_list:

    print(f"Number of spikes: {spikes}")

    working_path: str = os.path.join(path, f"{spikes}")

    files = glob.glob("*.npz", root_dir=working_path)

    assert len(files) > 0

    number_of_pattern: int = 0
    for file_id in natsorted(files):
        temp = np.load(os.path.join(working_path, file_id))
        number_of_pattern += temp["labels"].shape[0]

    assert number_of_pattern > 0

    labels = np.zeros((number_of_pattern), dtype=np.int64)
    images = np.zeros(
        (
            number_of_pattern,
            temp["the_images"].shape[1],
            temp["the_images"].shape[2],
            temp["the_images"].shape[3],
        ),
        dtype=np.float32,
    )

    position: int = 0
    for file_id in natsorted(files):
        temp = np.load(os.path.join(working_path, file_id))
        assert temp["labels"].shape[0] == temp["the_images"].shape[0]
        labels[position : position + temp["labels"].shape[0]] = temp["labels"]
        images[position : position + temp["labels"].shape[0], :, :, :] = temp[
            "the_images"
        ]
        position += temp["labels"].shape[0]

    images /= images.sum(axis=1, keepdims=True) + 1e-20

    np.savez_compressed(
        working_path + f"_{number_of_pattern}.npz", labels=labels, images=images
    )
