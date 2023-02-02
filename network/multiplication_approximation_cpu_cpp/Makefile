include ../.env
export

name = MultiplicationApproximation
type = CPU

PYPOSTFIX := $(shell $(PYBIN)python3-config --extension-suffix)
PYBIND11INCLUDE := $(shell $(PYBIN)python3 -m pybind11 --includes)
PARAMETERS_O = $(PARAMETERS_O_CPU) $(PYBIND11INCLUDE) 
PARAMETERS_Linker = $(PARAMETERS_Linker_CPU)

so_file = Py$(name)$(type)$(PYPOSTFIX)
pyi_file = Py$(name)$(type).pyi

all: ../$(so_file)


$(O_DIRS)error_term.o: \
		error_term.h \
		error_term.cpp
	mkdir -p $(O_DIRS) 
	$(CC) $(PARAMETERS_O) -c error_term.cpp -o $(O_DIRS)error_term.o


$(O_DIRS)approximation_multiplication_function.o: \
		approximation_multiplication_function.h \
		approximation_multiplication_function.cpp \
		error_term.h
	mkdir -p $(O_DIRS) 
	$(CC) $(PARAMETERS_O) -c approximation_multiplication_function.cpp -o $(O_DIRS)approximation_multiplication_function.o


$(O_DIRS)$(name)$(type).o: \
		$(name)$(type).h \
		$(name)$(type).cpp \
		approximation_multiplication_function.h
	mkdir -p $(O_DIRS) 
	$(CC) $(PARAMETERS_O) -c $(name)$(type).cpp -o $(O_DIRS)$(name)$(type).o

$(O_DIRS)Py$(name)$(type).o: $(name)$(type).h Py$(name)$(type).cpp 
	mkdir -p $(O_DIRS)
	$(CC) $(PARAMETERS_O) -c Py$(name)$(type).cpp -o $(O_DIRS)Py$(name)$(type).o

../$(so_file): \
		$(O_DIRS)$(name)$(type).o \
		$(O_DIRS)Py$(name)$(type).o \
		$(O_DIRS)approximation_multiplication_function.o \
		$(O_DIRS)error_term.o
	$(CC) $(PARAMETERS_Linker) -o ../$(so_file) \
		$(O_DIRS)$(name)$(type).o \
		$(O_DIRS)Py$(name)$(type).o \
		$(O_DIRS)approximation_multiplication_function.o \
		$(O_DIRS)error_term.o


#######################
clean:
	rm -rf $(O_DIRS)
	rm -f ../$(so_file)
	rm -f ../$(pyi_file)

