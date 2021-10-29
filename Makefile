ann_mnist_digits: ann_mnist_digits.cpp
	g++ ann_mnist_digits.cpp -g -o ann_mnist_digits -std=c++11  -larmadillo -lblas -Bstatic -Iarmadillo-10.6.2/include/ -Larmadillo-10.6.2/build

ann_mnist_digits_cuda: ann_mnist_digits.cu
	nvcc -O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -std=c++11 -g -Iarmadillo-10.6.2/include/  -I /usr/local/cuda-9.2/targets/x86_64-linux/include -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/ ann_mnist_digits.cu -L /usr/local/cuda-10.0/targets/x86_64-linux/lib -l nvblas -l lapack_static  -o ann_mnist_digits_cuda

	#nvcc -O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -std=c++11 -Iarmadillo-10.6.2/include/  -I /usr/local/cuda-9.2/targets/x86_64-linux/include -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/ ann_mnist_digits.cu -L /usr/local/cuda-10.0/targets/x86_64-linux/lib/liblapack_static.a -l nvblas -l lapack_static  -o ann_mnist_digits_cuda #-Xcompiler -static


# set some defaults.  ?= means don't override the value if it was set already
MPICXX?=mpic++
CXX?=g++
CXXFLAGS?=-std=c++11  -O2 -Bstatic
NVCC?=nvcc
NVFLAGS?=-O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -std=c++11
SHELL:=/usr/bin/bash
HOST=$(shell hostname)
# all targets
TARGETS = ann_mnist_digits ann_mnist_digits_cuda

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies

# all targets depend on the helper programs
$(TARGETS) : ann_mnist_digits_cuda ann_mnist_digits

LIBS_cuda   = -lnvblas -larmadillo -Larmadillo-10.6.2/build
INCFLAGS =  -Iarmadillo-10.6.2/include/


# wildcard rules
%_mpi.o : %_mpi.cpp
ifeq ($(HOSTNAME),fac-login-0.local) 
	module load gnu/7.2.0 gnutools mpi/openmpi3_eth; \
	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@
else 
	module load mpi/openmpi-x86_64;  \
	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@
endif 

%_mpi : %_mpi.cpp
#ifeq ($(HOSTNAME),fac-login-0.local) 
	module load gnu/7.2.0 gnutools mpi/openmpi3_eth;  \
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@
#else 
#	module load mpi/openmpi-x86_64; \
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@
#endif

%.o : %.cu
ifeq ($(HOSTNAME),fac-login-0.local) 
	module load cuda
else 
	module load cuda/10.1 gcc
endif
	$(NVCC) $(NVFLAGS) $(INCFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
#ifeq ($(HOSTNAME),fac-login-0.local) 
#	module load cuda
#else 
#	module load cuda/10.1 gcc
#endif
	$(NVCC) $(NVFLAGS) $(INCFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^)  $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
