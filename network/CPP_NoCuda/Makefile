# Change to your python bin directory (tested with Python 3.10.4)
PYBIN=~/P3.10GPU/bin/
CC=/usr/lib64/ccache/clang++

PYBIND11INCLUDE=`$(PYBIN)python3 -m pybind11 --includes`
PARAMETERS_O= -O3 -std=c++14 $(PYBIND11INCLUDE) -fPIC -Wall -fopenmp=libomp

PARAMETERS_Linker=-shared -lm -lomp -lstdc++ -Wall

PYPOSTFIX=`$(PYBIN)python3-config --extension-suffix`


all: PyHDynamicCNNManyIP \
		PySpikeGeneration2DManyIP \
		PyMultiApp 

#######################

HDynamicCNNManyIP.o: HDynamicCNNManyIP.h HDynamicCNNManyIP.cpp
	$(CC) $(PARAMETERS_O) -c HDynamicCNNManyIP.cpp -o HDynamicCNNManyIP.o

PyHDynamicCNNManyIP.o: HDynamicCNNManyIP.h PyHDynamicCNNManyIP.cpp 
	$(CC) $(PARAMETERS_O) -c PyHDynamicCNNManyIP.cpp -o PyHDynamicCNNManyIP.o

PyHDynamicCNNManyIP: HDynamicCNNManyIP.o PyHDynamicCNNManyIP.o
	$(CC) $(PARAMETERS_Linker) -o PyHDynamicCNNManyIP HDynamicCNNManyIP.o PyHDynamicCNNManyIP.o
	cp PyHDynamicCNNManyIP PyHDynamicCNNManyIP$(PYPOSTFIX)
	$(PYBIN)python3 pybind11_auto_pyi.py

#######################

SpikeGeneration2DManyIP.o: SpikeGeneration2DManyIP.h SpikeGeneration2DManyIP.cpp
	$(CC) $(PARAMETERS_O) -c SpikeGeneration2DManyIP.cpp -o SpikeGeneration2DManyIP.o

PySpikeGeneration2DManyIP.o: SpikeGeneration2DManyIP.h PySpikeGeneration2DManyIP.cpp 
	$(CC) $(PARAMETERS_O) -c PySpikeGeneration2DManyIP.cpp -o PySpikeGeneration2DManyIP.o

PySpikeGeneration2DManyIP: SpikeGeneration2DManyIP.o PySpikeGeneration2DManyIP.o
	$(CC) $(PARAMETERS_Linker) -o PySpikeGeneration2DManyIP SpikeGeneration2DManyIP.o PySpikeGeneration2DManyIP.o
	cp PySpikeGeneration2DManyIP PySpikeGeneration2DManyIP$(PYPOSTFIX)
	$(PYBIN)python3 pybind11_auto_pyi.py

#######################

MultiApp.o: MultiApp.h MultiApp.cpp approximation_multiplication_function.cpp \
                error_term.cpp 
	$(CC) $(PARAMETERS_O) -c MultiApp.cpp -o MultiApp.o

PyMultiApp.o: MultiApp.h PyMultiApp.cpp
	$(CC) $(PARAMETERS_O) -c PyMultiApp.cpp -o PyMultiApp.o

PyMultiApp: MultiApp.o PyMultiApp.o
	$(CC) $(PARAMETERS_Linker)  -o PyMultiApp MultiApp.o PyMultiApp.o
	cp PyMultiApp PyMultiApp$(PYPOSTFIX)
	$(PYBIN)python3 pybind11_auto_pyi.py

#######################
clean:
	rm -f PyHDynamicCNNManyIP
	rm -f PySpikeGeneration2DManyIP
	rm -f PyMultiApp
	rm -f *.o
	rm -f *.so

