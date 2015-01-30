/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_cuDNNException_hpp
#define maxdnn_cuDNNException_hpp

#include "cudnn.h"
#include "maxdnn/Exception.hpp"

#define MAXDNN_CUDNN_CHECK(R) {			\
        cudnnStatus_t status=(R);               \
        if (status!=CUDNN_STATUS_SUCCESS)       \
            throw cuDNNException(status);       \
    }
   
namespace maxdnn
{
    class cuDNNException : public Exception
    {
    public:

        cuDNNException(int cudnnError);
        
        std::string getDescription() const;
        
    private:

        int _cudnnError;
    };
}


#endif
