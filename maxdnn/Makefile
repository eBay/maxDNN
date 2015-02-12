# Template copied from: http://stackoverflow.com/questions/2481269/how-to-make-simple-c-makefile
CC=gcc
CXX=g++
RM=rm -f
AR=ar

# Compile kernels for Maxwell architecture.
export maxdnn_maxwell=1

#
# Set the locations of CUDA, cuDNN, and UnitTest++.
#
EXTERNALS=$(HOME)/develop/externals
CUDA_PATH=/usr/local/cuda
CUDNN_PATH=$(EXTERNALS)/cudnn-6.5-linux-x64-v2-rc2
UNITTEST_PATH=/usr/local/include/UnitTest++
MAXAS_PATH=$(EXTERNALS)/maxas

INCLUDE=-I.. -I$(UNITTEST_PATH) -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -I. -I$(CUDNN_PATH)

# DEBUG=2 => CPP and CUDA debugging.
ifeq ($(DEBUG),2)
CPPFLAGS=-g $(INCLUDE) -Wall
NVCCFLAGS=-G -g -lineinfo -Xcompiler $(INCLUDE)
else
ifeq ($(DEBUG),1)
# DEBUG=1 => CPP debugging, CUDA optimized
CPPFLAGS=-g $(INCLUDE) -Wall
NVCCFLAGS=-O3 -lineinfo -Xcompiler $(INCLUDE)
else
# default => CPP and CUDA optimized.
CPPFLAGS=-O3 -g $(INCLUDE) -Wall -Werror
NVCCFLAGS=-O3 -lineinfo -Xcompiler $(INCLUDE)
endif
endif
LDFLAGS=-g -L. -L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH)
LDLIBS=-lmaxdnn -lcuda -lcudart -lcudnn_static -lculibos -lnvToolsExt -lcublas -ldl -lpthread
TEST_LDFLAGS=$(LDFLAGS) -L$(UNITTEST_PATH)

NVCC=$(CUDA_PATH)/bin/nvcc

# maxdnn_maxwell=1 => GPU is Maxwell architecture.
ifeq ($(maxdnn_maxwell),1)
CUDA_ARCH=-gencode arch=compute_50,code=sm_50
else
# default => kepler support only
CUDA_ARCH=-gencode arch=compute_35,code=sm_35
endif

SRCS=\
	MemoryMap.cpp\
	FileSystem.cpp\
	FileName.cpp\
	File.cpp\
	CppData.cpp\
	TensorIO.cpp\
	MappedBlockData.cpp\
	Shape.cpp\
	Exception.cpp\
	Process.cpp\
	GpuException.cpp\
	Data.cpp\
	GpuData.cpp\
	HostData.cpp\
	convolution.cpp\
	Texture.cpp\
	cuDNNContext.cpp\
	convolution_cudnn.cpp\
	multiconvolution_64.cpp\
	profile.cpp

CUDA_SRCS=

CUDA_HEADERS=\
	IndexesInstance.cuh

TESTS=\
	test.cpp\
	maxdnn_test.cpp\
	maxdnn_test_main.cpp\
	FileName_test.cpp\
	FileSystem_test.cpp\
	File_test.cpp\
	Random_test.cpp\
	Shape_test.cpp\
	Tensor_test.cpp\
	MemoryMap_test.cpp\
	TensorIO_test.cpp\
	Data_test.cpp\
	convolution_test.cpp\
	cuDNNContext_test.cpp

CUDA_TEST_SRCS=\
	conv.cu

OBJS=$(subst .cpp,.o,$(SRCS))
CUDA_OBJS=$(subst .cu,.cuo,$(CUDA_SRCS))
CUDA_TEST_OBJS=$(subst .cu,.cuo,$(CUDA_TEST_SRCS))
TEST_OBJS=$(subst .cpp,.o,$(TESTS))

all: maxdnn_test.bin

# definition of network layers used in unit test.
MAXDNN_CONV_TEST_CONFIG=../networks/conv.cfg


# See http://www.gnu.org/savannah-checkouts/gnu/automake/manual/html_node/Multiple-Outputs.html
conv.cu:../scripts/ConvBuilder.py $(MAXDNN_CONV_TEST_CONFIG)
	python ../scripts/ConvBuilder.py $(MAXDNN_CONV_TEST_CONFIG) conv.cu conv.hpp conv_test.cpp

conv.hpp conv_test.cpp maxdnn/conv.hpp : conv.cu
## Recover from the removal of $@
	if test -f $(notdir $@); then :; else \
	  rm -f conv.cu; \
	  $(MAKE) conv.cu; \
        fi

maxdnn_test.bin: $(TEST_OBJS) $(CUDA_TEST_OBJS) maxdnn/conv.hpp libmaxdnn.a multiconvolution_64.cubin
	g++ $(TEST_LDFLAGS) -o maxdnn_test.bin $(TEST_OBJS) $(CUDA_TEST_OBJS) $(LDLIBS) -lUnitTest++ 

libmaxdnn.a: $(OBJS) $(CUDA_OBJS)
	$(AR) cr libmaxdnn.a $(OBJS) $(CUDA_OBJS)

%.cuo: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@

depend: .depend

multiconvolution_64.cubin: multiconvolution_64.cu multiconvolution_64.sass
	$(NVCC) -arch sm_50 -cubin $(INCLUDE) multiconvolution_64.cu
	$(MAXAS_PATH)/maxas.pl -e -k multiconvolution_64 multiconvolution_64.cubin multiconvolution_64-skeleton.asm
	$(MAXAS_PATH)/maxas.pl -i multiconvolution_64.sass multiconvolution_64.cubin || (echo "maxas.pl failed $$?"; exit 1)
	$(MAXAS_PATH)/maxas.pl -e -k multiconvolution_64 multiconvolution_64.cubin multiconvolution_64-final.asm

# Make sure generated header file, conv.hpp, is generated before
# depencencies are analyzed.
.depend: $(SRCS) $(TESTS) conv.cu
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)
	$(RM) $(TEST_OBJS)
	$(RM) $(CUDA_OBJS)
	$(RM) $(CUDA_TEST_OBJS)
	$(RM) libmaxdnn.a
	$(RM) maxdnn_test.bin
	$(RM) conv.cu
	$(RM) conv.hpp
	$(RM) conv_test.cpp
	$(RM) .depend
	$(RM) multiconvolution_64.cubin

dist-clean: clean
	$(RM) *~ .depend

include .depend
