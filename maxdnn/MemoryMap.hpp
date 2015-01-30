/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_MemoryMap_h
#define maxdnn_MemoryMap_h

#include <stdint.h>
#include <string>

namespace maxdnn
{
    class MemoryMap
    {
    public:

        enum Protection
        {
            Read=1,
            Write=2
        };

        enum Flags
        {
            Shared=1,
            Private=2,
            Anonymous=4,
            HugeTable=8,
            Locked=16,
            Nonblock=32,
            Populate=64
        };

        MemoryMap();
        
        MemoryMap(int fd, size_t length, int protection, int flags, size_t offset = 0);

        ~MemoryMap();

        bool map(int fd, size_t length, int protection, int flags, size_t offset = 0);
        
        bool unmap();

        const uint8_t *getMemory() const { return _ptr; }
        
        uint8_t *getMemory() { return _ptr; }

        size_t getLength() const { return _length; } 

        bool isOk() const;
        
private:

        uint8_t *_map(int fd, size_t length, int protection, int flags, size_t offset);
        
        uint8_t *_ptr;
        size_t _length;
    };
}


#endif
