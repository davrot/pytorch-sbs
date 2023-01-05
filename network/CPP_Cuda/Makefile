# Change to your python bin directory (tested with Python 3.10.4)
PYBIN=~/P3.10GPU/bin/
NVCC=/usr/local/cuda-12/bin/nvcc -allow-unsupported-compiler
CC=/usr/lib64/ccache/clang++

PYBIND11INCLUDE=`$(PYBIN)python3 -m pybind11 --includes`
PARAMETERS_O= -O3 -std=c++14 $(PYBIND11INCLUDE) -ccbin=$(CC) \
                                -Xcompiler "-fPIC -Wall -fopenmp=libomp"

PARAMETERS_Linker=-Xcompiler "-shared -lm -lomp -lstdc++ -Wall"

PYPOSTFIX=`$(PYBIN)python3-config --extension-suffix`


all: PyHDynamicCNNManyIP \
		PySpikeGeneration2DManyIP \
		PyMultiApp 

#######################

HDynamicCNNManyIP.o: HDynamicCNNManyIP.h HDynamicCNNManyIP.cu 
	$(NVCC) $(PARAMETERS_O) -c HDynamicCNNManyIP.cu -o HDynamicCNNManyIP.o

PyHDynamicCNNManyIP.o: HDynamicCNNManyIP.h PyHDynamicCNNManyIP.cpp 
	$(NVCC) $(PARAMETERS_O) -c PyHDynamicCNNManyIP.cpp -o PyHDynamicCNNManyIP.o

PyHDynamicCNNManyIP: HDynamicCNNManyIP.o PyHDynamicCNNManyIP.o
	$(NVCC) $(PARAMETERS_Linker) -o PyHDynamicCNNManyIP HDynamicCNNManyIP.o PyHDynamicCNNManyIP.o
	cp PyHDynamicCNNManyIP PyHDynamicCNNManyIP$(PYPOSTFIX)
	$(PYBIN)python3 pybind11_auto_pyi.py

#######################

SpikeGeneration2DManyIP.o: SpikeGeneration2DManyIP.h SpikeGeneration2DManyIP.cu
	$(NVCC) $(PARAMETERS_O) -c SpikeGeneration2DManyIP.cu -o SpikeGeneration2DManyIP.o

PySpikeGeneration2DManyIP.o: SpikeGeneration2DManyIP.h PySpikeGeneration2DManyIP.cpp 
	$(NVCC) $(PARAMETERS_O) -c PySpikeGeneration2DManyIP.cpp -o PySpikeGeneration2DManyIP.o

PySpikeGeneration2DManyIP: SpikeGeneration2DManyIP.o PySpikeGeneration2DManyIP.o
	$(NVCC) $(PARAMETERS_Linker) -o PySpikeGeneration2DManyIP SpikeGeneration2DManyIP.o PySpikeGeneration2DManyIP.o
	cp PySpikeGeneration2DManyIP PySpikeGeneration2DManyIP$(PYPOSTFIX)
	$(PYBIN)python3 pybind11_auto_pyi.py


#######################

MultiApp.o: MultiApp.h MultiApp.cu approximation_multiplication_function.cpp \
                gpu_approximation_multiplication_function.cu error_term.cpp gpu_error_term.cu
	$(NVCC) $(PARAMETERS_O) -c MultiApp.cu -o MultiApp.o

PyMultiApp.o: MultiApp.h PyMultiApp.cpp
	$(NVCC) $(PARAMETERS_O) -c PyMultiApp.cpp -o PyMultiApp.o

PyMultiApp: MultiApp.o PyMultiApp.o
	$(NVCC) $(PARAMETERS_Linker)  -o PyMultiApp MultiApp.o PyMultiApp.o
	cp PyMultiApp PyMultiApp$(PYPOSTFIX)
	$(PYBIN)python3 pybind11_auto_pyi.py

#######################
clean:
	rm -f PyHDynamicCNNManyIP
	rm -f PySpikeGeneration2DManyIP
	rm -f PyMultiApp
	rm -f *.o
	rm -f *.so

