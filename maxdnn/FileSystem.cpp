/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#define _XOPEN_SOURCE 500
#include "maxdnn/FileName.hpp"
#include <maxdnn/FileSystem.hpp>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <ftw.h>
#include <iostream>
using namespace std;

namespace
{
    int _delTree(const char *fpath, const struct stat *sb,
                 int typeflag, struct FTW *ftwbuf);
}

namespace maxdnn
{
    namespace FileSystem
    {
        bool remove(const string &fileName)
        {
            return ::unlink(fileName.c_str()) == 0;
        }
        
        bool exists(const string &fileName)
        {
            return ::access(fileName.c_str(), F_OK) == 0;
        }
        
        bool readable(const string &fileName)
        {
            return ::access(fileName.c_str(), R_OK) == 0;
        }

        bool writeable(const string &fileName)
        {
            return ::access(fileName.c_str(), W_OK) == 0;
        }

        bool mkpath(const string &pathName)
        {
            if (exists(pathName)) {
                return isDirectory(pathName);
            }
            string parent = FileName(pathName).getParent();
            if (parent.empty()) {
                return true;
            }
            if (!mkpath(parent)) {
                return false;
            }
            return mkdir(pathName);
        }

        bool mkdir(const string &fileName)
        {
            return ::mkdir(fileName.c_str(), 0777) != -1;
        }

        bool rmdir(const string &fileName)
        {
            return ::rmdir(fileName.c_str()) != -1;
        }
        
        bool deltree(const string &fileName)
        {
            return nftw(fileName.c_str(),
                        _delTree,
                        16,
                        FTW_DEPTH|FTW_PHYS) == 0;
        }

        bool touch(const std::string &fileName)
        {
            int fd = ::open(fileName.c_str(), O_CREAT, 0777);
            if (fd != -1) {
                close(fd);
                return true;
            }
            return false;
        }
        
        off_t getFileSize(const string &fileName)
        {
            off_t sz = 0;
            struct stat st;
            if (stat(fileName.c_str(), &st) != -1) {
                sz = st.st_size;
            }
            return sz;
        }

        bool isRegularFile(const string &fileName)
        {
            bool is = false;
            struct stat st;
            if (stat(fileName.c_str(), &st) != -1) {
                is = S_ISREG(st.st_mode);
            }
            return is;
        }
                
        bool isDirectory(const string &fileName)
        {
            bool is = false;
            struct stat st;
            if (stat(fileName.c_str(), &st) != -1) {
                is = S_ISDIR(st.st_mode);
            }
            return is;
        }

        void sync()
        {
            ::sync();
        }
        
    }
}

namespace
{
    int _delTree(const char *fpath, const struct stat *sb,
                 int typeflag, struct FTW *ftwbuf)
    {
        switch(typeflag) {
        case FTW_F:
        case FTW_SL:
        case FTW_SLN:
            ::unlink(fpath);
            break;
        case FTW_D:
        case FTW_DP:
            ::rmdir(fpath);
            break;
        default:
            // The path cannot be read or the stat called failed
            // on it. Better not touch it.
            break;
        }
            
        return FTW_CONTINUE;
    }
        
}
