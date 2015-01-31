/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/GpuData.hpp"
#include "maxdnn/HostData.hpp"
#include "maxdnn/GpuException.hpp"
#include "maxdnn/gpu.h"
#include <cuda_runtime.h>
#include <sstream>

namespace maxdnn
{
    GpuData::Pointer GpuData::_proto(new GpuData);

    GpuData::GpuData()
        : _data(0),
          _size(0)
    {
    }
    
    GpuData::GpuData(size_t size) throw (GpuAllocException)
    {
        _size = size;
        _data = 0;
        if (cudaMalloc(&_data, _size) != cudaSuccess) {
            throw GpuAllocException(size);
        }
    }

    GpuData::~GpuData()
    {
        if (_data) {
            cudaFree(_data);
        }
    }

    void GpuData::copy(const Data &src)
    {
        src.copyTo(*this);
    }
    
    void GpuData::copyTo(HostData &dst) const
    {
        copyToHost(dst.getData(), dst.getSize());
    }
    
    void GpuData::copyTo(GpuData &dst) const
    {
        copyToDevice(dst.getData(), dst.getSize());
    }

    void GpuData::copyFromHost(const void *src, size_t size)
    {
        MAXDNN_ASSERT(size <= _size, BoundsException(size, _size, "GpuData_copyFromHost_size"));
        CUDA_CHECK(cudaMemcpy(_data, src, size, cudaMemcpyHostToDevice));
    }

    void GpuData::copyToHost(void *dst, size_t size) const
    {
        MAXDNN_ASSERT(size <= _size, BoundsException(size, _size, "GpuData_copyToHost_size"));
        CUDA_CHECK(cudaMemcpy(dst, _data, size, cudaMemcpyDeviceToHost));
    }

    void GpuData::copyToDevice(void *dst, size_t size) const
    {
        MAXDNN_ASSERT(size <= _size, BoundsException(size, _size, "GpuData_copyToDevice_size"));
        CUDA_CHECK(cudaMemcpy(dst, _data, size, cudaMemcpyDeviceToDevice));
    }
}
