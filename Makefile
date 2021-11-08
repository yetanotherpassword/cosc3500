# set some defaults.  ?= means don't override the value if it was set already
CXX?=g++
CXXFLAGS?=-std=c++11 -g

NVCC?=nvcc
NVFLAGS?=-g --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets
NVFLAGS_ann_mnist_digits_serial= -DSERIAL_ONLY
# all targets
TARGETS=ann_mnist_digits_cuda ann_mnist_digits_serial
LIB= -Larmadillo-10.6.2/build/ -larmadillo -llapack_static
INC= -Iarmadillo-10.6.2/include/

ann_mnist_digits_serial: ann_mnist_digits.cu
	$(NVCC) $(INC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@
ann_mnist_digits_cuda: ann_mnist_digits.cu
	$(NVCC) $(INC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

# wildcard rules - compile source code to object files and executables

%.o : %.cpp
	$(XX) $(CXXFLAGS) $(CXXFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cu
	$(NVCC) $(INC) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
