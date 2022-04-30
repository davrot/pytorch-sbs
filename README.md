# pytorch-sbs
SbS Extension for PyTorch


# Based on these scientific papers

**Back-Propagation Learning in Deep Spike-By-Spike Networks**  
David Rotermund and Klaus R. Pawelzik  
Front. Comput. Neurosci., https://doi.org/10.3389/fncom.2019.00055  
https://www.frontiersin.org/articles/10.3389/fncom.2019.00055/full  

**Efficient Computation Based on Stochastic Spikes**  
Udo Ernst, David Rotermund, and Klaus Pawelzik  
Neural Computation (2007) 19 (5): 1313–1343. https://doi.org/10.1162/neco.2007.19.5.1313  
https://direct.mit.edu/neco/article-abstract/19/5/1313/7183/Efficient-Computation-Based-on-Stochastic-Spikes  

# Python

It was programmed with 3.10.4. And I used some 3.10 Python expression. Thus you might get problems with older Python versions. 

# C++

It works without compiling the C++ modules. However it is 10x slower.   
You need to modify the Makefile in the C++ directory to your Python installation.  
In addition yoir Python installation needs the PyBind11 package installed. You might want to perform a  
pip install pybind11  
The Makefile uses clang as a compiler. If you want something else then you need to change the Makefile.
The SbS.py autodetectes if the required C++ .so modules are in the same directory as the SbS.py file.  

# SbS layer class

## Variables

```
epsilon_xy  
```

```
epsilon_0  
```

```
epsilon_t  
```

```
weights  
```

```
kernel_size  
```

```
stride  
```

```
dilation  
```

```
padding  
```

```
output_size  
```

```
number_of_spikes  
```

```
number_of_cpu_processes  
```

```
number_of_neurons  
```

```
number_of_input_neurons  
```

```
h_initial  
```

```
alpha_number_of_iterations  
```

## Constructor 
```
def **__init__**(  
    self,  
    number_of_input_neurons: int,  
    number_of_neurons: int,  
    input_size: list[int],  
    forward_kernel_size: list[int],  
    number_of_spikes: int,  
    epsilon_t: torch.Tensor,  
    epsilon_xy_intitial: float = 0.1,  
    epsilon_0: float = 1.0,  
    weight_noise_amplitude: float = 0.01,  
    is_pooling_layer: bool = False,  
    strides: list[int] = [1, 1],  
    dilation: list[int] = [0, 0],  
    padding: list[int] = [0, 0],  
    alpha_number_of_iterations: int = 0,  
    number_of_cpu_processes: int = 1,  
) -> None:
```

## Methods

```
def **initialize_weights**(  
    self,  
    is_pooling_layer: bool = False,  
    noise_amplitude: float = 0.01,  
) -> None:  
```
For the generation of the initital weights. Switches between normal initial random weights and pooling weights.

---

```
def **initialize_epsilon_xy**(  
    self,  
    eps_xy_intitial: float,  
) -> None:  
```
Creates initial epsilon xy matrices.

---
```
def **set_h_init_to_uniform**(self) -> None:  
```

---
```
def **backup_epsilon_xy**(self) -> None:  
def **restore_epsilon_xy**(self) -> None:  
def **backup_weights(self)** -> None:  
def **restore_weights(self)** -> None:  
```

---
```
def **threshold_epsilon_xy**(self, threshold: float) -> None:  
def **threshold_weights**(self, threshold: float) -> None:  
```

---
```
def **mean_epsilon_xy**(self) -> None:  
```

---
```
def **norm_weights**(self) -> None:
```

# Parameters in JSON file

data_mode: str = field(default="")  
data_path: str = field(default="./")  

batch_size: int = field(default=500)  

learning_step: int = field(default=0)  
learning_step_max: int = field(default=10000)  

number_of_cpu_processes: int = field(default=-1)  

number_of_spikes: int = field(default=0)  
cooldown_after_number_of_spikes: int = field(default=0)  

weight_path: str = field(default="./Weights/")  
eps_xy_path: str = field(default="./EpsXY/")  
    
reduction_cooldown: float = field(default=25.0)  
epsilon_0: float = field(default=1.0)  

update_after_x_batch: float = field(default=1.0)  


## network_structure (required!)
Parameters of the network. The details about its layers and the number of output neurons.  

number_of_output_neurons: int = field(default=0)  
forward_neuron_numbers: list[list[int]] = field(default_factory=list)  
is_pooling_layer: list[bool] = field(default_factory=list)  

forward_kernel_size: list[list[int]] = field(default_factory=list)  
strides: list[list[int]] = field(default_factory=list)  
dilation: list[list[int]] = field(default_factory=list)  
padding: list[list[int]] = field(default_factory=list)  

w_trainable: list[bool] = field(default_factory=list)  
eps_xy_trainable: list[bool] = field(default_factory=list)  
eps_xy_mean: list[bool] = field(default_factory=list)  


## learning_parameters
Parameter required for training   

learning_active: bool = field(default=True)  

loss_coeffs_mse: float = field(default=0.5)  
loss_coeffs_kldiv: float = field(default=1.0)  

optimizer_name: str = field(default="Adam")  
learning_rate_gamma_w: float = field(default=-1.0)  
learning_rate_gamma_eps_xy: float = field(default=-1.0)  
learning_rate_threshold_w: float = field(default=0.00001)  
learning_rate_threshold_eps_xy: float = field(default=0.00001)  

lr_schedule_name: str = field(default="ReduceLROnPlateau")  
lr_scheduler_factor_w: float = field(default=0.75)  
lr_scheduler_patience_w: int = field(default=-1)  

lr_scheduler_factor_eps_xy: float = field(default=0.75)  
lr_scheduler_patience_eps_xy: int = field(default=-1)  

number_of_batches_for_one_update: int = field(default=1)  
overload_path: str = field(default="./Previous")  

weight_noise_amplitude: float = field(default=0.01)  
eps_xy_intitial: float = field(default=0.1)  

test_every_x_learning_steps: int = field(default=50)  
test_during_learning: bool = field(default=True)  

alpha_number_of_iterations: int = field(default=0)  

## augmentation
Parameters used for data augmentation.  

crop_width_in_pixel: int = field(default=2)  

flip_p: float = field(default=0.5)  

jitter_brightness: float = field(default=0.5)  
jitter_contrast: float = field(default=0.1)  
jitter_saturation: float = field(default=0.1)  
jitter_hue: float = field(default=0.15)  


## ImageStatistics (please ignore)
(Statistical) information about the input. i.e. mean values and the x and y size of the input  

mean: list[float] = field(default_factory=list)  
the_size: list[int] = field(default_factory=list)  
