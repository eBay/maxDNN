/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/MemoryMap.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace
{
    int linuxProtection(int protection);
    int linuxFlags(int flags);
}

namespace maxdnn
{
    MemoryMap::MemoryMap()
    {
        _length = 0;
        _ptr = 0;
    }
    
    MemoryMap::MemoryMap(int fd, size_t length, int protection, int flags, size_t offset)
    {
        _length = 0;
        _ptr = 0;

        map(fd, length, protection, flags, offset);
    }

    MemoryMap::~MemoryMap()
    {
        unmap();
    }

    bool MemoryMap::map(int fd, size_t length, int protection, int flags, size_t offset)
    {
        unmap();
        _length = length;
        if (fd >= 0 && length > 0) {
            _ptr = static_cast<uint8_t *>(::mmap(NULL,
                                                 _length,
                                                 linuxProtection(protection),
                                                 linuxFlags(flags),
                                                 fd,
                                                 offset));
        }
        return isOk();
    }
    
    bool MemoryMap::unmap()
    {
        bool r = true;
        if (_length > 0 && _ptr != MAP_FAILED) {
            r = (::munmap(_ptr, _length) == 0);
            _length = 0;
            _ptr = 0;
        }
        return r;
    }
    
    bool MemoryMap::isOk() const
    {
        return _ptr != MAP_FAILED;
    }
}

namespace
{
    bool isset(int flags, int flag)
    {
        return ((flags&flag)==flag);
    }
    

    using namespace maxdnn;
    
#define MAP_FLAG(FLAGS_OUT, FLAGS_IN, FROM, TO)\
        if (isset(FLAGS_IN, FROM)) {\
            FLAGS_OUT |= TO;        \
                                    \
        }

    int linuxProtection(int protection) 
    {
        int flags_out =  0;
        MAP_FLAG(flags_out, protection, MemoryMap::Read, PROT_READ);
        MAP_FLAG(flags_out, protection, MemoryMap::Write, PROT_WRITE);
        return flags_out;
    }
    
    int linuxFlags(int flags)
    {
        int flags_out = 0;
        MAP_FLAG(flags_out, flags, MemoryMap::Shared, MAP_SHARED);
        MAP_FLAG(flags_out, flags, MemoryMap::Private, MAP_PRIVATE);
        MAP_FLAG(flags_out, flags, MemoryMap::Anonymous, MAP_ANONYMOUS);
        MAP_FLAG(flags_out, flags, MemoryMap::HugeTable, MAP_HUGETLB);
        MAP_FLAG(flags_out, flags, MemoryMap::Locked, MAP_LOCKED);
        MAP_FLAG(flags_out, flags, MemoryMap::Nonblock, MAP_NONBLOCK);
        MAP_FLAG(flags_out, flags, MemoryMap::Populate, MAP_POPULATE);
        return flags_out;
    }
}
