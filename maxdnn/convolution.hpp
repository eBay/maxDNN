/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/Tensor.hpp"
#include "cudnn.h"

namespace maxdnn
{
    void convolve_gpu(const Tensor<Float> &input,
                      const Tensor<Float> &kernels,
                      Tensor<Float> &output,
                      unsigned stride,
                      unsigned padding);

    void convolve_cpu(const Tensor<Float> &input,
                      const Tensor<Float> &kernels,
                      Tensor<Float> &output,
                      unsigned stride,
                      unsigned padding,
                      int maxPixels=0);

    void convolve_cuDNN(cudnnHandle_t handle,
                        const Tensor<Float> &input,
                        const Tensor<Float> &kernels,
                        Tensor<Float> &output,
                        unsigned stride,
                        unsigned padding,
                        const char *profileName);

    void convolve_ccn2(const Tensor<Float> &input,
                        const Tensor<Float> &kernels,
                        Tensor<Float> &output,
                        unsigned stride,
                        unsigned padding);
}
