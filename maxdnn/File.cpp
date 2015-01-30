/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/File.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace maxdnn
{
    File::File()
    {
        _fd = -1;
    }

    File::File(const std::string &fileName, int flags)
    {
        _fd = -1;
        open(fileName, flags);
    }
    

    File::~File()
    {
        close();
    }

    bool File::open(const std::string &fileName, int flags)
    {
        close();

        _fileName = fileName;
        
        int f = 0;
        if ((flags & (Read|Write))==(Read|Write)) {
            f |= O_RDWR;
        } else if ((flags & Read)==Read) {
            f |= O_RDONLY;
        } else if ((flags & Write)==Write) {
            f |= O_WRONLY;
        } else {
            // Read or Write must be set.
            return false;
        }
        if (flags & Create) {
            f |= O_CREAT;
        }
        if (flags & Append) {
            f |= O_APPEND;
        }
        
        _fd = ::open(_fileName.c_str(), f, 0777);

        return _fd != -1;
    }
    
    bool File::close()
    {
        bool r = true;
        if (_fd != -1) {
            r = ::close(_fd) == 0;
            _fd = -1;
            _fileName.clear();
        }
        return r;
    }
    
    bool File::write(const void *data, size_t len)
    {
        return _fd != -1 && size_t(::write(_fd, data, len)) == len;
    }
    
    bool File::read(void *data, size_t len)
    {
        return _fd != -1 && size_t(::read(_fd, data, len)) == len;
    }
 
        
    off64_t File::seek(off64_t offset, Whence whence)
    {
        int w;
        switch(whence) {
        default:
        case Set:
            w = SEEK_SET;
            break;
        case Current:
            w = SEEK_CUR;
            break;
        case End:
            w = SEEK_END;
            break;
        }
        return ::lseek64(_fd, offset, w);
    }
    
    off64_t File::tell()
    {
        return ::lseek64(_fd, 0, SEEK_CUR);
    }
    
    bool File::fsync()
    {
        return _fd == -1 || ::fsync(_fd) == 0;
    }
}
