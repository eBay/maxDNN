/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/MemoryMap.hpp"
#include "maxdnn/FileSystem.hpp"
#include "maxdnn/File.hpp"
#include "UnitTest++.h"
#include <vector>
#include <cstring>
#include <stdint.h>
using namespace maxdnn;
using namespace std;
namespace fs=maxdnn::FileSystem;

struct MemoryMapTestFixture
{
    MemoryMapTestFixture()
    {
        fileName = "/tmp/maxdnn-memory-map-test-file";
        length = 16385;
    }

    ~MemoryMapTestFixture()
    {
    }
    
    string fileName;
    size_t length;
};
    
SUITE(MemoryMap)
{
    TEST_FIXTURE(MemoryMapTestFixture, CreateWriteRead)
    {
        fs::remove(fileName);

        {
            vector<uint8_t> data(length);
            for (size_t i = 0; i < length; ++i) {
                data[i] = uint8_t(i%256);
            }
            
            File fwrite(fileName, File::Write|File::Create);
            CHECK(fwrite.isOk());
            CHECK(fwrite.write(&data[0], length));
        }
        
        {
            File file(fileName, File::Read);
            MemoryMap mread;
            CHECK(mread.map(file.getDescriptor(), length, MemoryMap::Read, MemoryMap::Private));
            uint8_t *data = static_cast<uint8_t *>(mread.getMemory());
            for (size_t i = 0; i < length; ++i) {
                CHECK_EQUAL(uint8_t(i%256), data[i]);
            }
        }
    }
}
