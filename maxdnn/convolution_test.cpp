/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/maxdnn_test.hpp"
#include "maxdnn/convolution.hpp"
#include "maxdnn/conv.hpp"
#include "maxdnn/TensorOperations.hpp"
#include "maxdnn/GpuData.hpp"
#include "maxdnn/MappedBlockData.hpp"
#include "maxdnn/File.hpp"
#include "maxdnn/TensorIO.hpp"
#include "maxdnn/Texture.hpp"
#include "maxdnn/FileSystem.hpp"
#include "maxdnn/Process.hpp"
#include "maxdnn/profile.hpp"
#include "cudnn.h"
#include "maxdnn/cuDNNException.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
using namespace maxdnn;
using namespace std;
using namespace tr1;

struct ConvolutionFixture
{
    ConvolutionFixture()
    {
        Cooldown = getCooldown();
        Seed = 2354;
        Tolerance = 1.e-2;
        NumIters = getConvIters();
        w_min = -1;
        w_max = 1;
        input_min = -1;
        input_max = 1;

        // Ensure the test data directory exists.
        FileSystem::mkpath(getTestDataDirectory());
    }

    virtual ~ConvolutionFixture()
    {
    }
    
    typedef void (*conv_func)(const Tensor<Float> &, const Tensor<Float> &, Tensor<Float> &);

    void cooldown()
    {
        if (Cooldown > 0) {
            cout << Cooldown << " sec cooldown .. ";
            cout.flush();
            Process::sleep(Cooldown);
            cout << "done." << endl;
        }
    }

    void testConvolution(conv_func conv)
    {
        Random ran(Seed);

        cout << "Running test " << UnitTest::CurrentTest::Details()->testName << endl;

        // TODO: make config parameter
        const int padding = 5;
                
        Shape inputShape(NumColors, H_in, W_in, BatchSize,
                         BatchSize,
                         (W_in+padding)*BatchSize,
                         (H_in+padding)*(W_in+padding)*BatchSize);
            
        Shape filterShape(NumColors, KernelSize, KernelSize, NumFilters);

        Tensor<Float> input(inputShape);
        Tensor<Float> filters(filterShape);
        input.fillWithZeros();
        setUniformRandom(input, ran, input_min, input_max);
        setUniformRandom(filters, ran, w_min, w_max);

        if (!noTest() && (generateReferenceOutput() || !FileSystem::exists(ResultFileName))) {
            cout << "Generating reference output " << ResultFileName << " .. ";
            cout.flush();
            Tensor<Float> output_cpu(NumFilters, H_out, W_out, BatchSize);
            convolve_cpu(input, filters, output_cpu, Stride, Padding);
            CHECK(write(output_cpu, ResultFileName));
            FileSystem::sync();
            cout << "done." << endl;
            cooldown();            
        }

        for (int i=0; i<NumIters; ++i) {
            Tensor<Float> input_gpu(input, GpuData::prototype());
            Tensor<Float> filters_gpu(filters, GpuData::prototype());
            Tensor<Float> output_gpu(NumFilters, H_out, W_out, BatchSize, GpuData::prototype());
            (*conv)(input_gpu, filters_gpu, output_gpu);
            CUDA_CHECK(cudaDeviceSynchronize());
            cooldown();

            if (i == NumIters-1) {
                Tensor<Float> output_host(output_gpu, CpuData::prototype());
                if (!noTest()) {
                    cout << "Checking output of test " << UnitTest::CurrentTest::Details()->testName << " .. ";
                    Tensor<Float> output_cpu;
                    read(output_cpu, ResultFileName);
                    checkRelClose(output_cpu, output_host, Tolerance);
                    cout << "done." << endl;
                    cooldown();
                }
            }
        }
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
    
    void testConvolution_cuDNN(const char *profileName)
    {
        // cuDNN does not honor the output dimensions we request for
        // alexnet layer conv1, so we skip the verifiction of the
        // output. This also means we slightly over-estimate the speed
        // of cuDNN on this layer, because its output width is
        // actually 1 pixel shorter.
        
        const char *NoTest = "convolve_cudnn_alexnet_conv1";

        
        cout << "Running test " << UnitTest::CurrentTest::Details()->testName << endl;

        Random ran(Seed);

        Tensor<Float> input(NumColors, H_in, W_in, BatchSize);
        Tensor<Float> filters(NumColors, KernelSize, KernelSize, NumFilters);
        setUniformRandom(input, ran, input_min, input_max);
        setUniformRandom(filters, ran, w_min, w_max);

        if (!noTest() &&
            strncmp(UnitTest::CurrentTest::Details()->testName, NoTest, strlen(NoTest)) &&
            (generateReferenceOutput() || !FileSystem::exists(ResultFileName))) {
            cout << "Generating reference output " << ResultFileName << " .. ";
            cout.flush();
            Tensor<Float> output_cpu(NumFilters, H_out, W_out, BatchSize);
            convolve_cpu(input, filters, output_cpu, Stride, Padding);
            CHECK(write(output_cpu, ResultFileName));
            FileSystem::sync();
            cout << "done." << endl;
            cooldown();
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

            convolve_cuDNN(context, input_gpu, filters_gpu, output_gpu, Stride, Padding, profileName);
            CUDA_CHECK(cudaDeviceSynchronize());
            cooldown();

            if (i == NumIters-1) {
                cout.flush();
                Tensor<Float> output_host(output_gpu, CpuData::prototype());
                if (!noTest() &&
                    strncmp(UnitTest::CurrentTest::Details()->testName, NoTest, strlen(NoTest))) {
                    cout << "Checking output of test " << UnitTest::CurrentTest::Details()->testName << " .. ";
                    cout.flush();
                    Tensor<Float> output_cpu;
                    read(output_cpu, ResultFileName);
                    Tensor<Float> output_cpu_cudnn(BatchSize, NumFilters, H_out, W_out);
                    CHWN_to_NCHW(output_cpu, output_cpu_cudnn);
                    checkRelClose(output_cpu_cudnn, output_host, Tolerance);
                    cout << "done." << endl;
                    cooldown();
                }
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
    int NumIters;
    int Cooldown;
};

// Include the test source file that is generated by ConvBuilder.py.
#include "conv_test.cpp"

