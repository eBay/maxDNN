/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "cudnn.h"
#include "maxdnn/convolution.hpp"
#include "maxdnn/cuDNNException.hpp"
#include "maxdnn/GpuData.hpp"
#include "maxdnn/profile.hpp"
#include "maxdnn/TensorOperations.hpp"
#include <stdint.h>

namespace maxdnn
{
    void convolve_cuDNN(cudnnHandle_t handle,
                        const Tensor<Float> &input,
                        const Tensor<Float> &kernels,
                        Tensor<Float> &output,
                        unsigned stride,
                        unsigned padding,
                        const char *profileName)
    {
        cudnnTensorDescriptor_t input_h;
        cudnnFilterDescriptor_t kernels_h;
        cudnnTensorDescriptor_t output_h;
        cudnnConvolutionDescriptor_t conv_h;

        float alpha = 1.0f;
        float beta = 0.0f;

        MAXDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_h));
        MAXDNN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_h, 
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      input.getShape().K,
                                                      input.getShape().L,
                                                      input.getShape().M,
                                                      input.getShape().N));

        MAXDNN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernels_h));
        MAXDNN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernels_h, 
                                                      CUDNN_DATA_FLOAT,
                                                      kernels.getShape().K,
                                                      kernels.getShape().L,
                                                      kernels.getShape().M,
                                                      kernels.getShape().N));

        MAXDNN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_h));
        MAXDNN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_h, 
                                                      CUDNN_TENSOR_NCHW,
                                                      CUDNN_DATA_FLOAT,
                                                      output.getShape().K,
                                                      output.getShape().L,
                                                      output.getShape().M,
                                                      output.getShape().N));

        MAXDNN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_h));
        MAXDNN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_h,
                                                           padding,
                                                           padding,
                                                           stride,
                                                           stride,
                                                           1,
                                                           1,
                                                           CUDNN_CROSS_CORRELATION));
        
        int n, c, h, w;
        MAXDNN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_h,
                                                                 input_h,
                                                                 kernels_h,
                                                                 &n,
                                                                 &c,
                                                                 &h,
                                                                 &w));

        MAXDNN_ASSERT_SIZE_MATCH(output.getShape().K, n, "n");
        MAXDNN_ASSERT_SIZE_MATCH(output.getShape().L, c, "c");
        // TODO: cudnn generates sizes incompatible with some networks
        // specifications,
        // such as Alexnet conv1.
        //  MAXDNN_ASSERT_SIZE_MATCH(output.getShape().M, h, "h");
        // MAXDNN_ASSERT_SIZE_MATCH(output.getShape().N, w, "w");

        cudnnConvolutionFwdAlgo_t algo;
        MAXDNN_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                               input_h,
                                                               kernels_h,
                                                               conv_h,
                                                               output_h,
                                                               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//                                                               CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                                                               0,
                                                               &algo));

        size_t workspaceNumBytes = 0;
        MAXDNN_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                                   input_h,
                                                                   kernels_h,
                                                                   conv_h,
                                                                   output_h,
                                                                   algo,
                                                                   &workspaceNumBytes));

        Tensor<uint8_t> workspace(1, 1, 1, workspaceNumBytes, GpuData::prototype());

        MAXDNN_SCOPE(profileName);
        {
            MAXDNN_CUDNN_CHECK(cudnnConvolutionForward(handle,
                                                       &alpha,
                                                       input_h,
                                                       &input(),
                                                       kernels_h,
                                                       &kernels(),
                                                       conv_h,
                                                       algo,
                                                       &workspace(),
                                                       workspace.getSize(),
                                                       &beta,
                                                       output_h,
                                                       &output()));
        }
        
        MAXDNN_CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_h));
        MAXDNN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_h));
        MAXDNN_CUDNN_CHECK(cudnnDestroyFilterDescriptor(kernels_h));
        MAXDNN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_h));
    }
}
