/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_gpu_h
#define maxdnn_gpu_h

#include "maxdnn/GpuException.hpp"

#ifdef __CUDACC__
#define MAXDNN_METHOD __device__ __host__
#else
#define MAXDNN_METHOD
#endif

#define CUDA_CHECK(R)                                   \
  {                                                     \
    int r = (R);                                        \
    MAXDNN_ASSERT(r==cudaSuccess, CudaException(r));       \
  }

#endif
