/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_CppData_h
#define maxdnn_CppData_h

#include "maxdnn/HostData.hpp"
#include <stdint.h>

namespace maxdnn
{
    class CppData : public HostData
    {
    public:

        typedef std::tr1::shared_ptr<CppData> Pointer;

        static Pointer &prototype() 
        {
            return _proto;
        }
        
        Data::Pointer clone(size_t size) const
        {
            return make(size);
        }
        
        static Pointer make(size_t size)
        {
            return Pointer(new CppData(size));
        }
        
        ~CppData()
        {
            delete[] _data;
        }

        const void *getData() const { return _data; }

        void *getData() { return _data; }

        size_t getSize() const { return _size; }
        
        bool lock();
        
        bool unlock();
        
    private:

        CppData();
                
        CppData(size_t size)
            : _data(new uint8_t[size]),
              _size(size)
        {
        }

	uint8_t *_data;
	size_t _size;

        static Pointer _proto;
    };

    typedef CppData CpuData;
}


#endif
