/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_HostData_h
#define maxdnn_HostData_h

#include "maxdnn/Data.hpp"

namespace maxdnn
{
    class HostData : public Data
    {
    public:

        typedef std::tr1::shared_ptr<HostData> Pointer;
        
        virtual Data::Pointer clone(size_t size) const = 0;

        virtual ~HostData() {}

        virtual void *getData() = 0;

        virtual const void *getData() const = 0;

        virtual size_t getSize() const = 0;

        void copy(const Data &src) { src.copyTo(*this); } 

        void copyTo(HostData &dst) const;

        void copyTo(GpuData &dst) const;

        bool operator==(const HostData &other) const;
    };
}


#endif
