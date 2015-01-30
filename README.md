maxDNN
======

High Efficiency Convolution Kernel for NVIDIA Maxwell GPU Architecture

Introduction
------------

maxDNN provides a high efficiency convolution kernel for the forward propagation phase of convolutional neural networks. The kernel run on NVIDIA Maxwell GPUs (eg, Geforce GTX980). The kernel is a derivative work of the SGEMM kernel from the Maxas Maxwell Assembler project.

Technical details and performance analysis of the maxDNN kernel are documented in the following technical report:

"maxDNN: An Efficient Convolution Kernel for Deep Learning with Maxwell GPUs"
http://arxiv-web3.library.cornell.edu/abs/1501.06633


Requirements
------------

+ NVIDIA GPU of compute capability 5.0 or better (i.e., Maxwell GPU)
+ Linux (tested with Ubuntu 12.04)
+ CUDA Toolkit (tested with version 6.5): https://developer.nvidia.com/cuda-downloads
+ UnitTest++ unit testing framework: http://unittest-cpp.sourceforge.net/
+ cuDNN library version 2: https://developer.nvidia.com/cuDNN
+ MaxAs Assembler for NVIDIA Maxwell Architecture: https://code.google.com/p/maxas/

Install
-------

1. Get the software.

    git clone https://the/repo/url/maxDNN.git

2. Install the external requirements above.

3. Set up the MaxAs environment by following the instructions at: https://code.google.com/p/maxas/wiki/GettingStarted
The instructions are for MS Windows, but using the Linux analog of each of the path variables works fine.

4. Edit maxdnn/Makefile to ensure that CUDA_PATH, CUDNN_PATH, UNITTEST_PATH, and MAXAS_PATH are correct for your system.

5. Build.

    cd maxDNN/maxdnn
    make all

6. Run the convolution suite unit tests:

    maxdnn/maxdnn_test.bin suite convolution

or run all unit tests

    maxdnn/maxdnn_test.bin

7. Optionally, run the tests again, checking result accuracy against the result of CPU convolution. Do so by setting the path where generated reference data should be stored. Generating reference data will take about 30 minutes, because the CPU convolution is very slow, but subsequent runs will just read it from disk instead of generating it again.

    export maxdnn_test_data=/path/to/where/I/want/to/store/test/data

    maxdnn/maxdnn_test.bin suite convolution
