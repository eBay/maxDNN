/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/cuDNNContext.hpp"
#include "maxdnn/cuDNNException.hpp"
using namespace std;

namespace
{
    const char *getcuDNNErrorString(cudnnStatus_t status)
    {
        switch(status) {
        case CUDNN_STATUS_SUCCESS:
            return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED:
            return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED:
            return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM:
            return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR:
            return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE:
            return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH:
            return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR:
            return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED:
            return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED:
            return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR:
            return "CUDNN_STATUS_LICENSE_ERROR";
        default:
            return "Unknown error.";
        }
    }
}


namespace maxdnn
{
    cuDNNException::cuDNNException(int cudnnError)
        : _cudnnError(cudnnError)
    {
    }
        
    string cuDNNException::getDescription() const
    {
        return getcuDNNErrorString(static_cast<cudnnStatus_t>(_cudnnError));
    }
    
    cuDNNContext::cuDNNContext()
        : _handle(0)
    {
    }
    
    cuDNNContext::~cuDNNContext()
    {
        destroy();
    }

    void cuDNNContext::create()
    {
        MAXDNN_CUDNN_CHECK(cudnnCreate(&_handle));
    }

    void cuDNNContext::destroy()
    {
        if (_handle != 0) {
            cudnnDestroy(_handle);
            _handle = 0;
        }
    }
}
