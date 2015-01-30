/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/test.hpp"
#include "cudnn.h"
#include "maxdnn/cuDNNException.hpp"
#include "maxdnn/convolution.hpp"
#include "maxdnn/TensorOperations.hpp"
#include "maxdnn/GpuData.hpp"
#include "maxdnn/MappedBlockData.hpp"
#include "maxdnn/File.hpp"
#include "maxdnn/TensorIO.hpp"
#include <cuda_runtime.h>
#include <algorithm>
using namespace maxdnn;
using namespace std;
using namespace tr1;

// Disable test only when doing speed analysis experiments.
#define TEST_ENABLED 1

struct ConvolutionFixture_cuDNN
{
    ConvolutionFixture_cuDNN()
    {
        Seed = 2354;
        Tolerance = 1.e-3;
        GenerateReferenceOutput = false;
        NumIters = 1;
    }
    
    // Input order is CHWN
    static void CHWN_to_NCHW(const Tensor<Float> &in, Tensor<Float> &out)
    {
        const int C = in.getShape().K;
        const int H = in.getShape().L;
        const int W = in.getShape().M;
        const int N = in.getShape().N;

        for (int i=0; i<N; ++i) {
            for (int c=0; c<C; ++c) {
                for (int y=0; y<H; ++y) {
                    for (int x=0; x<W; ++x) {
                        out(i,c,y,x) = in(c,y,x,i);
                    }
                }
            }
        }
    }

    static void CHWK_to_KCHW(const Tensor<Float> &in, Tensor<Float> &out)
    {
        const int C = in.getShape().K;
        const int H = in.getShape().L;
        const int W = in.getShape().M;
        const int K = in.getShape().N;
        
        for (int k=0; k<K; ++k) {
            for (int c=0; c<C; ++c) {
                for (int y=0; y<H; ++y) {
                    for (int x=0; x<W; ++x) {
                        out(k,c,y,x) = in(c,y,x,k);
                    }
                }
            }
        }
    }
    
    void testConvolution()
    {
        Random ran(Seed);

        Tensor<Float> input(NumColors, H_in, W_in, BatchSize);
        Tensor<Float> filters(NumColors, KernelSize, KernelSize, NumFilters);
        setUniformRandom(input, ran, input_min, input_max);
        setUniformRandom(filters, ran, w_min, w_max);

        if (GenerateReferenceOutput) {
            Tensor<Float> output_cpu(NumFilters, H_out, W_out, BatchSize);
            convolve_cpu(input, filters, output_cpu, Stride, Padding);
            CHECK(write(output_cpu, ResultFileName));
        }

        Tensor<Float> input_cudnn(BatchSize, NumColors, H_in, W_in);
        Tensor<Float> filters_cudnn(NumFilters, NumColors, KernelSize, KernelSize);

        CHWN_to_NCHW(input, input_cudnn);
        CHWK_to_KCHW(filters, filters_cudnn);

        cudnnHandle_t context;
        MAXDNN_CUDNN_CHECK(cudnnCreate(&context));

        for (int i=0; i<NumIters; ++i) {
            Tensor<Float> input_gpu(input_cudnn, GpuData::prototype());
            Tensor<Float> filters_gpu(filters_cudnn, GpuData::prototype());
            Tensor<Float> output_gpu(BatchSize, NumFilters, H_out, W_out, GpuData::prototype());

            convolve_cuDNN(context, input_gpu, filters_gpu, output_gpu, Stride, Padding);

            if (i == NumIters-1) {
                Tensor<Float> output_host(output_gpu, CpuData::prototype());
                Tensor<Float> output_cpu;
                read(output_cpu, ResultFileName);
                Tensor<Float> output_cpu_cudnn(BatchSize, NumFilters, H_out, W_out);
                CHWN_to_NCHW(output_cpu, output_cpu_cudnn);
#if TEST_ENABLED
                checkRelClose(output_cpu_cudnn, output_host, Tolerance);
#endif
            }
        }

        MAXDNN_CUDNN_CHECK(cudnnDestroy(context));
    }

    unsigned Stride;
    unsigned Padding;
    unsigned KernelSize;
    unsigned NumFilters;
    unsigned BatchSize;
    unsigned NumColors;
    unsigned W_in;
    unsigned H_in;
    unsigned W_out;
    unsigned H_out;
    Float w_min;
    Float w_max;
    Float input_min;
    Float input_max;
    unsigned Seed;
    string ResultFileName;
    Float Tolerance;
    bool GenerateReferenceOutput;
    int NumIters;
};

struct Convolution_Alexnet_Conv1_cuDNN : public ConvolutionFixture_cuDNN
{
    Convolution_Alexnet_Conv1_cuDNN()
    {
        Padding = 0;
        Stride = 4;
        KernelSize = 11;
        NumFilters = 64;
        BatchSize = 128;
        NumColors = 3;
        W_in = 227;
        H_in = 227;
        W_out = 55;
        H_out = 55;

        w_min = -1;
        w_max = 1;
        input_min = -1;
        input_max = 1;

        ResultFileName = getTestFile("convolve_cpu.erd");
    }
};


struct Convolution_Alexnet_Conv4_cuDNN : public ConvolutionFixture_cuDNN
{
    Convolution_Alexnet_Conv4_cuDNN()
    {
        Padding = 1;
        Stride = 1;
        KernelSize = 3;
        NumFilters = 256;
        BatchSize = 128;
        NumColors = 384;
        W_in = 13;
        H_in = 13;
        W_out = 13;
        H_out = 13;

        w_min = -1;
        w_max = 1;
        input_min = -1;
        input_max = 1;

        ResultFileName = getTestFile("convolve_alexnet_conv4_cpu.erd");
        Tolerance = 1.e-2;
    }
};

SUITE(cuDNN)
{
    TEST_FIXTURE(Convolution_Alexnet_Conv1_cuDNN, convolve_alexnet_conv1)
    {
        testConvolution();
    }

    TEST_FIXTURE(Convolution_Alexnet_Conv4_cuDNN, convolve_alexnet_conv4)
    {
        testConvolution();
    }
}
