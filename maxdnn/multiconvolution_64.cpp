/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/multiconvolution_64.hpp"
#include "maxdnn/ConvolutionIndexes.hpp"
#include "maxdnn/ConvolutionBlockings.hpp"
#include "maxdnn/GpuData.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

namespace
{
    CUmodule hModule;
    CUfunction hKernel;
    CUtexref texInput, texFilters;
    CUevent hStart, hStop;
    maxdnn::ConvolutionIndexes indexes;
    maxdnn::ConvolutionBlockings blockings;
}

namespace maxdnn
{
    void multiconvolution_64_unload()
    {
        // Cleanup and shutdown of cuda
        CUDA_CHECK( cuEventDestroy(hStart) );
        CUDA_CHECK( cuEventDestroy(hStop) );
        CUDA_CHECK( cuModuleUnload(hModule) );
    }

    void multiconvolution_64(const Tensor<Float> &input_gpu, const Tensor<Float> &filters_gpu, Tensor<Float> &output_gpu, int stride, int padding, float alpha, int numIterations)
    {
        int Ni = input_gpu.getShape().K;
        int Nb = input_gpu.getShape().N;
        int Ho = output_gpu.getShape().L;
        int Wo = output_gpu.getShape().M;
        int No = filters_gpu.getShape().N;
        int Sk = filters_gpu.getShape().L;
        int ldc = Wo*Ho*Nb;
        int Nbf = (No+63)/64;
        int Wstride4 = input_gpu.getShape().strideL/4;
        int Nld = 4;

        int blocksX = Wo;
        int blocksY = Ho;
        int blocksZ = ((No+63)/64)*((Nb+63)/64);
        int threads = 64;

        // int whammy = 32;
        // Tensor<int32_t> diagnostic(blocksZ, blocksY, blocksX, threads*whammy);
        // Tensor<int32_t> diagnostic_gpu(diagnostic, GpuData::prototype());
        Tensor<int32_t> diagnostic;
        Tensor<int32_t> diagnostic_gpu;

        // Load our kernel
        if (hModule == 0) {
            CUDA_CHECK( cuModuleLoad(&hModule, "multiconvolution_64.cubin") );
            CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "multiconvolution_64") );
            // Load the textures
            CUDA_CHECK( cuModuleGetTexRef(&texInput, hModule, "texInput") );
            CUDA_CHECK( cuModuleGetTexRef(&texFilters, hModule, "texFilters") );
            // Configure the textures
            CUDA_CHECK( cuTexRefSetFormat(texInput, CU_AD_FORMAT_FLOAT, 4) );
            CUDA_CHECK( cuTexRefSetFormat(texFilters, CU_AD_FORMAT_FLOAT, 4) );
            CUDA_CHECK( cuEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) );
            CUDA_CHECK( cuEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );
        }
    
        CUDA_CHECK( cuTexRefSetAddress(NULL, texInput, (CUdeviceptr)&input_gpu(), input_gpu.getSize()) );
        CUDA_CHECK( cuTexRefSetAddress(NULL, texFilters, (CUdeviceptr)&filters_gpu(), filters_gpu.getSize()) );

        // Set the constant memory
        int indexesOffset;
        int numIndexes;
        {
            int position = -1;
            if (indexes.addIndexesForParameters(input_gpu.getShape(), Ni, Sk, Nld, position)) {
                size_t size;
                CUdeviceptr indexes_gpu;
                CUDA_CHECK(cuModuleGetGlobal(&indexes_gpu, &size, hModule, "multiconvolution_64_Indexes"));
                MAXDNN_ASSERT(size == sizeof(ConvolutionIndexesGpu), 
                           maxdnn::BoundsException(size, sizeof(ConvolutionIndexesGpu), "indexesSize"));
                CUDA_CHECK(cuMemcpyHtoD(indexes_gpu, &indexes, sizeof(ConvolutionIndexesGpu)));
            }
            indexesOffset = indexes.arrayOffsets[position];
            numIndexes = indexes.arraySizes[position];
        }

        int blockingOffset;
        {
            int position = -1;
            if (blockings.addBlockingsForParameters(No, Nb, position)) {
                size_t size;
                CUdeviceptr blockings_gpu;
                CUDA_CHECK(cuModuleGetGlobal(&blockings_gpu, &size, hModule, "multiconvolution_64_Blockings"));
                MAXDNN_ASSERT(size == sizeof(ConvolutionBlockingsGpu), 
                           maxdnn::BoundsException(size, sizeof(ConvolutionBlockingsGpu), "blockingsSize"));
                CUDA_CHECK(cuMemcpyHtoD(blockings_gpu, &blockings, sizeof(ConvolutionBlockingsGpu)));
            }
            blockingOffset = blockings.arrayOffsets[position];
        }
        
        // Setup the params
        int negPadding = -padding;

        float *out = &output_gpu();
        int *diag = &diagnostic_gpu();
        void* params[] = { &out, &indexesOffset, &numIndexes, &stride, &Nb, &Nbf, &Wstride4, &Wo, &Ho, &No, &ldc, &negPadding, &alpha, &blockingOffset, &diag};

        float totalTime = 0;
        float ms = 0;

        // Launch the kernel

        for (int i=0; i<numIterations; ++i) {
            CUDA_CHECK( cuEventRecord(hStart, NULL) );
            CUDA_CHECK( cuLaunchKernel(hKernel, blocksX, blocksY, blocksZ, threads, 1, 1, 0, 0, params, 0) );
            CUDA_CHECK( cuEventRecord(hStop, NULL) );
            CUDA_CHECK( cuEventSynchronize(hStop) );
            CUDA_CHECK( cuEventElapsedTime(&ms, hStart, hStop) );
            totalTime += ms;
        }
        
	
//        cout << "Elapsed time = " << ms << " totalTime" << " in " << numIterations << " iterations" << endl;

//        diagnostic.copy(diagnostic_gpu);
//        int *diag_cpu = &diagnostic();
//        (void)diag_cpu;
    }
}
