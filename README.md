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

# Parameters

