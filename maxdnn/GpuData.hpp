/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_GpuData_h
#define maxdnn_GpuData_h

#include "maxdnn/Data.hpp"
#include "maxdnn/GpuException.hpp"
#include <stdint.h>

namespace maxdnn
{
    class GpuData : public Data
    {
    public:

        typedef std::tr1::shared_ptr<GpuData> Pointer;

        static GpuData::Pointer prototype() 
        {
            return _proto;
        }
        
        Data::Pointer clone(size_t size) const
        {
            return make(size);
        }

        static Pointer make(size_t size) throw (GpuAllocException)
        {
            return Pointer(new GpuData(size));
        }
        
        ~GpuData();

        const void *getData() const { return _data; }

        void *getData() { return _data; }

        size_t getSize() const { return _size; }

        void copy(const Data &src);

        void copyTo(HostData &dst) const;

        void copyTo(GpuData &dst) const;

        void copyFromHost(const void *src, size_t size);

        void copyToHost(void *dst, size_t size) const;
        
        void copyToDevice(void *dst, size_t size) const;

    private:

        GpuData();

        GpuData(size_t size) throw (GpuAllocException);
        
	uint8_t *_data;
	size_t _size;

        static GpuData::Pointer _proto;
    };
}

#endif
