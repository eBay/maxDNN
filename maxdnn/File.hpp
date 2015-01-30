/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_File_h
#define maxdnn_File_h

#include <string>

namespace maxdnn
{
    class File
    {
    public:

        enum Flags { Read=1, Write=2, Create=4, Append=8 };

        File();

        File(const std::string &fileName, int flags);

        ~File();

        bool open(const std::string &fileName, int flags);

        bool close();

        bool write(const void *data, size_t len);
        
        bool read(void *data, size_t len);
        
        enum Whence { Set, Current, End};

        off64_t seek(off64_t offset, Whence whence);
        
        off64_t tell();

        bool fsync();

        bool isOk() const { return _fd != -1; } 

        int getDescriptor() { return _fd; } 

    private:

        std::string _fileName;
        int _fd;
    };
}


#endif
