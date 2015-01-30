/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/CppData.hpp"
#include <sys/mman.h>

namespace maxdnn
{
    CppData::Pointer CppData::_proto(new CppData);

    CppData::CppData()
        : _data(0),
          _size(0)
    {
    }

    bool CppData::lock()
    {
        return mlock(_data, _size) == 0;
    }
    
    bool CppData::unlock()
    {
        return munlock(_data, _size) == 0;
    }
}
