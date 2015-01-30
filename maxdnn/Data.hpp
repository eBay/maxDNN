/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Data_h
#define maxdnn_Data_h

#include <tr1/memory>

namespace maxdnn
{
    class HostData;
    class GpuData;
    
    class Data
    {
    public:

        typedef std::tr1::shared_ptr<Data> Pointer;
        
        virtual ~Data() {}

        virtual Pointer clone(size_t size) const = 0;

        virtual void *getData() = 0;

        virtual const void *getData() const = 0;

        virtual size_t getSize() const = 0;

        virtual void copy(const Data &src) = 0;

        virtual void copyTo(HostData &dst) const = 0;

        virtual void copyTo(GpuData &dst) const = 0;
    };
}


#endif
