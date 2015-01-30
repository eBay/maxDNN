/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/GpuException.hpp"
#include <cuda_runtime.h>
#include <sstream>
using namespace std;

namespace maxdnn
{
    GpuAllocException::GpuAllocException(size_t size)
        : AllocException(size, "GPU")
    {
    } 

    GpuAllocException::~GpuAllocException() throw()
    {
    }

    CudaException::CudaException(int cudaError)
        : _cudaError(cudaError)
    {
    }
    
    string CudaException::getDescription() const
    {
        const char *s = cudaGetErrorString(cudaError_t(_cudaError));
        if (s == 0) {
            ostringstream serr;
            serr << "Unknown CUDA error " << _cudaError;
            return serr.str();
        }
        return s;
    }
}
