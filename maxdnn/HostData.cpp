/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/HostData.hpp"
#include "maxdnn/Exception.hpp"
#include "maxdnn/GpuData.hpp"
#include <cstring>

namespace maxdnn
{
    void HostData::copyTo(HostData &dst) const
    {
        if (dst.getSize() < getSize()) {
            throw BoundsException(dst.getSize(), getSize(), "HostData_size");
        }
        
        memcpy(dst.getData(), getData(), getSize());
    }
    
    void HostData::copyTo(GpuData &dst) const
    {
        if (dst.getSize() < getSize()) {
            throw BoundsException(dst.getSize(), getSize(), "HostData_size");
        }

        dst.copyFromHost(getData(), getSize());
    }

    bool HostData::operator==(const HostData &other) const
    {
        return
            getSize() == other.getSize() &&
            memcmp(getData(), other.getData(), getSize()) == 0;
    }
}

