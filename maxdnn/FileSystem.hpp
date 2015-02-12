/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_FileSystem_h
#define maxdnn_FileSystem_h

#include <string>
#include <sys/types.h>

namespace maxdnn
{
    namespace FileSystem
    {
        bool remove(const std::string &fileName);
        
        bool exists(const std::string &fileName);

        bool readable(const std::string &fileName);

        bool writeable(const std::string &fileName);

        bool mkpath(const std::string &fileName);

        bool mkdir(const std::string &fileName);

        bool rmdir(const std::string &fileName);

        bool deltree(const std::string &fileName);

        bool touch(const std::string &fileName);
        
        off_t getFileSize(const std::string &fileName);

        bool isRegularFile(const std::string &fileName);
        
        bool isDirectory(const std::string &fileName);

        void sync();
    }
}


#endif
