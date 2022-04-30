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

# Change to your python bin directory (tested with Python 3.10.4)
PYBIN=~/P3.10/bin/
COMPILER=clang++

PYBIND11INCLUDE=`$(PYBIN)python3 -m pybind11 --includes`

PARAMETERS= -O3 -std=c++14 -fPIC  $(PYBIND11INCLUDE) -Wall -fopenmp=libomp


all: PyHDynamicCNNManyIP \
		PySpikeGeneration2DManyIP \

#######################

HDynamicCNNManyIP.o: HDynamicCNNManyIP.h HDynamicCNNManyIP.cpp 
	$(COMPILER) $(PARAMETERS) -c HDynamicCNNManyIP.cpp -o HDynamicCNNManyIP.o

PyHDynamicCNNManyIP.o: HDynamicCNNManyIP.h PyHDynamicCNNManyIP.cpp 
	$(COMPILER) $(PARAMETERS) -c PyHDynamicCNNManyIP.cpp -o PyHDynamicCNNManyIP.o

PyHDynamicCNNManyIP: HDynamicCNNManyIP.o PyHDynamicCNNManyIP.o
	$(COMPILER) -shared -o PyHDynamicCNNManyIP HDynamicCNNManyIP.o PyHDynamicCNNManyIP.o -lm -lomp -lstdc++ -Wall
	cp PyHDynamicCNNManyIP ../PyHDynamicCNNManyIP`$(PYBIN)python3-config --extension-suffix`

#######################

SpikeGeneration2DManyIP.o: SpikeGeneration2DManyIP.h SpikeGeneration2DManyIP.cpp 
	$(COMPILER) $(PARAMETERS) -c SpikeGeneration2DManyIP.cpp -o SpikeGeneration2DManyIP.o

PySpikeGeneration2DManyIP.o: SpikeGeneration2DManyIP.h PySpikeGeneration2DManyIP.cpp 
	$(COMPILER) $(PARAMETERS) -c PySpikeGeneration2DManyIP.cpp -o PySpikeGeneration2DManyIP.o

PySpikeGeneration2DManyIP: SpikeGeneration2DManyIP.o PySpikeGeneration2DManyIP.o
	$(COMPILER) -shared -o PySpikeGeneration2DManyIP SpikeGeneration2DManyIP.o PySpikeGeneration2DManyIP.o -lm -lomp -lstdc++ -Wall
	cp PySpikeGeneration2DManyIP ../PySpikeGeneration2DManyIP`$(PYBIN)python3-config --extension-suffix`


#######################

clean:
	rm -f PyHDynamicCNNManyIP
	rm -f PySpikeGeneration2DManyIP
	rm -f *.o

