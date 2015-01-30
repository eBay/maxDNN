/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_cuDNNContext_hpp
#define maxdnn_cuDNNContext_hpp

#include "cudnn.h"

namespace maxdnn
{
    class cuDNNContext
    {
    public:

        cuDNNContext();
        
        ~cuDNNContext();
        
        void create();
        
        void destroy();

    private:

        cudnnHandle_t _handle;
    };
}


#endif
