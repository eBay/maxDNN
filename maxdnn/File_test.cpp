/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/File.hpp"
#include "maxdnn/FileSystem.hpp"
#include "maxdnn/FileName.hpp"
#include "maxdnn/test.hpp"
#include "UnitTest++.h"
#include <vector>
#include <cstring>
#include <sys/types.h>
#include <stdint.h>
using namespace maxdnn;
using namespace std;
namespace fs=maxdnn::FileSystem;

struct FileTestFixture
{
    FileTestFixture()
    {
        FileName testDir = getTestDataDirectory();
        if (!testDir.isEmpty()) {
            fileName = testDir / "file-test-file";
        }
    }

    ~FileTestFixture()
    {
    }
    
    FileName fileName;
};
    
SUITE(File)
{
    TEST_FIXTURE(FileTestFixture, CreateWriteRead)
    {
        if (!fileName.isEmpty()) {
            
            File file(fileName.getString(), File::Create|File::Read|File::Write);

            CHECK(fs::exists(fileName.getString()));
            CHECK(fs::isRegularFile(fileName.getString()));

            const char *data = "Now is the time for all good objects to be dumped";
            const size_t n = strlen(data);
        
            vector<uint8_t> buf(n);
            CHECK(file.write(data, n));
            CHECK_EQUAL(0, file.seek(0, File::Set));
            CHECK(file.read(&buf[0], n));
        
            CHECK_EQUAL(0, memcmp(data, &buf[0], n));
        }
    }
}
