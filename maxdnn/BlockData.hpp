/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_BlockData_h
#define maxdnn_BlockData_h

#include "maxdnn/HostData.hpp"
#include "maxdnn/Shape.hpp"

namespace maxdnn
{
    /// A block of data with a tensor shape.
    class BlockData : public HostData
    {
    public:

        typedef std::tr1::shared_ptr<BlockData> Pointer;
        
        virtual Data::Pointer clone(size_t size) const = 0;

        virtual ~BlockData() {}

        virtual void *getData() = 0;

        virtual const void *getData() const = 0;

        const Shape& getShape() 
        {
            return _shape;
        }
        
    protected:

        BlockData()
            : _shape(0, 0, 0, 0)
        {
        }

        Shape _shape;
    };
}

#endif
