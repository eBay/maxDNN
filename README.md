#maxDNN

High Efficiency Convolution Kernel for NVIDIA Maxwell GPU Architecture

Introduction
------------

maxDNN is an existence proof that high efficiency convolution GPU kernels are possible. We solve the problem for the forward propagation phase of convolutional neural networks, and achieve roughly 95% computational efficiency for a typical network layer. This is a significant improvement over the state of the art which performs in the 30%-75% range.

The kernel runs on NVIDIA Maxwell GPUs (eg, Geforce GTX980) and is a derivative work of the SGEMM kernel from the [Maxas Maxwell Assembler project](https://code.google.com/p/maxas/).

Technical details and performance analysis of the maxDNN kernel are documented in the report
[maxDNN: An Efficient Convolution Kernel for Deep Learning with Maxwell GPUs](http://arxiv-web3.library.cornell.edu/abs/1501.06633)

Cite as:

```
@Article{Lavin:2015md,
     author    = "Andrew Lavin",
     title     = "{maxDNN: An Efficient Convolution Kernel for Deep Learning with Maxwell GPUs}",
     year      = "2015",
     archivePrefix = "arXiv",
     eprint        = "1501.06633",
     primaryClass  = "cs.NE",
}
```

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

        git clone https://github.com/eBay/maxDNN.git

2. Install the external requirements above.

3. Set up the MaxAs environment by following the instructions at: https://code.google.com/p/maxas/wiki/GettingStarted
The instructions are for MS Windows, but using the Linux analog of each of the path variables works fine.

4. Edit maxdnn/Makefile to ensure that CUDA_PATH, CUDNN_PATH, UNITTEST_PATH, and MAXAS_PATH are correct for your system.

5. Build.

        cd maxDNN/maxdnn
        make all

6. Run the convolution suite unit tests:

        maxdnn/maxdnn_test.bin suite convolution
        
7.  or run all unit tests:

        maxdnn/maxdnn_test.bin

8. Optionally, run the tests again, checking result accuracy against the result of CPU convolution. Do so by setting the path where generated reference data should be stored. Generating reference data will take about 30 minutes, because the CPU convolution is very slow, but subsequent runs will just read it from disk instead of generating it again.

        export maxdnn_test_data=/path/to/test/data/storage
        maxdnn/maxdnn_test.bin suite convolution

Benchmark
---------------
You can use nvprof to measure the efficiency of GPU kernels. The following sample gathers all the statistics that were used in the [tech report](http://arxiv-web3.library.cornell.edu/abs/1501.06633) to measure computational efficiency for Alexnet layer conv1:

```
nvprof -s --metrics flop_sp_efficiency,inst_fp_32,inst_integer,inst_bit_convert,inst_control,inst_misc,inst_executed,tex_cache_hit_rate,l2_tex_read_hit_rate --events elapsed_cycles_sm ./maxdnn_test.bin convolve_maxdnn_alexnet_conv1

==13380== NVPROF is profiling process 13380, command: ./maxdnn_test.bin convolve_maxdnn_alexnet_conv1
Running test convolve_maxdnn_alexnet_conv1
==13380== Warning: Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Success: 1 tests passed.
Test time: 87.08 seconds.
==13380== Profiling application: ./maxdnn_test.bin convolve_maxdnn_alexnet_conv1
==13380== Profiling result:
==13380== Event result:
Invocations                                Event Name         Min         Max         Avg
Device "GeForce GTX 980 (0)"
	Kernel: multiconvolution_64
         10                         elapsed_cycles_sm    75288048    75580076    75451292

==13380== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GTX 980 (0)"
	Kernel: multiconvolution_64
         10                             inst_executed                     Instructions Executed   320480600   320480600   320480600
         10                                inst_fp_32                   FP Instructions(Single)  9144115200  9144115200  9144115200
         10                              inst_integer                      Integer Instructions   240064000   240064000   240064000
         10                          inst_bit_convert                  Bit-Convert Instructions           0           0           0
         10                              inst_control                 Control-Flow Instructions    24006400    24006400    24006400
         10                                 inst_misc                         Misc Instructions    26329600    26329600    26329600
         10                        tex_cache_hit_rate                    Texture Cache Hit Rate      10.36%      10.74%      10.48%
         10                      l2_tex_read_hit_rate               L2 Hit Rate (Texture Reads)      84.56%      85.43%      85.09%
         10                        flop_sp_efficiency              FLOP Efficiency(Peak Single)      94.39%      94.76%      94.55%
```

