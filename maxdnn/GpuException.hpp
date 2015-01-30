/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_GpuException_hpp
#define maxdnn_GpuException_hpp

#include "maxdnn/Exception.hpp"

namespace maxdnn
{
    class GpuAllocException : public AllocException
    {
    public:

        GpuAllocException(size_t size);

        virtual ~GpuAllocException() throw();
    };

    class CudaException : public Exception
    {
    public:

        CudaException(int cudaError);

        std::string getDescription() const;

    private:

        int _cudaError;
    };
    
}

#endif
