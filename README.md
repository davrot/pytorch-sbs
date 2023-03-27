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

You need to modify the Makefile in the C++ directory to your Python installation.  

In addition your Python installation needs the PyBind11 package installed. You might want to perform a  
pip install pybind11  
The Makefile uses clang as a compiler. If you want something else then you need to change the Makefile.
For CUDA I used version 12.0.

# Config files and pre-existing weights

Three .json config files are required: 

dataset.json : Information about the dataset

network.json : Describes the network architecture

def.json : Controlls the other parameters 

If you want to load existing weights, just put them in a sub-folder called Previous

# Other relevant scientific papers

## NNMF

**Learning the parts of objects by non-negative matrix factorization**  
Lee, Daniel D., and H. Sebastian Seung. Nature 401.6755 (1999): 788-791.  
https://doi.org/10.1038/44565  

**Algorithms for non-negative matrix factorization.**  
Lee, Daniel, and H. Sebastian Seung. Advances in neural information processing systems 13 (2000).  
https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html  

## SbS
**Massively Parallel FPGA Hardware for Spike-By-Spike Networks**  
David Rotermund, Klaus R. Pawelzik  
https://doi.org/10.1101/500280  

**Biologically plausible learning in a deep recurrent spiking network**
David Rotermund, Klaus R. Pawelzik  
https://doi.org/10.1101/613471  

**Accelerating Spike-by-Spike Neural Networks on FPGA With Hybrid Custom Floating-Point and Logarithmic Dot-Product Approximation**  
Yarib Nevarez, David Rotermund, Klaus R. Pawelzik, Alberto Garcia-Ortiz  
https://doi.org/10.1109/access.2021.3085216  


# Tested installation (under Fedora 37)

mkdir PySource  
cd PySource  
wget https://www.python.org/ftp/python/3.11.2/Python-3.11.2.tgz  
tar -xvzf Python-3.11.2.tgz  
cd Python-3.11.2  
./configure --prefix=/home/[YOURUSERNAME]/P3.11 --enable-optimizations  
make -j 10  
make install  

cd /home/[YOURUSERNAME]/P3.11/bin  
./pip3 install --upgrade pip  
./pip3 install numpy scipy pandas flake8 pep8-naming black matplotlib seaborn ipython jupyterlab mypy dataclasses-json dataconf mat73 ipympl torch torchtext pywavelets scikit-image opencv-python scikit-learn tensorflow_datasets tensorboard tqdm argh sympy jsmin pybind11 pybind11-stubgen pigar asciichartpy torchvision torchaudio tensorflow natsort  

Please adapt the .env file in the network directory before compling the PyBind11 SbS modules. 
