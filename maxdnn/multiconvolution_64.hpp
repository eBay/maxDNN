/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef multiconvolution_64_hpp
#define multiconvolution_64_hpp

#include "maxdnn/Tensor.hpp"

namespace maxdnn
{
    void multiconvolution_64(const Tensor<Float> &input_gpu, const Tensor<Float> &filters_gpu, Tensor<Float> &output_gpu, int stride, int padding, float alpha, int numIterations=1);

    void multiconvolution_64_unload();
}

#endif
