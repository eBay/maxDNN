/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "UnitTest++.h"
#include "maxdnn/CppData.hpp"
#include "maxdnn/GpuData.hpp"
using namespace maxdnn;
using namespace std;
    
struct DataTestFixture
{
    DataTestFixture()
    {
        size = 8192;
        
        cppData = CppData::make(size);
        uint8_t *buf = static_cast<uint8_t *>(cppData->getData());
        
        for(size_t i = 0; i < size; ++i) {
            buf[i] = uint8_t(size%256);
        }
    }
    
    ~DataTestFixture()
    {
    }

    CppData::Pointer cppData;
    size_t size;
};

    
SUITE(Data)
{
    TEST_FIXTURE(DataTestFixture, GpuData_copy)
    {
        CHECK_EQUAL(size, cppData->getSize());
        GpuData::Pointer gpuData = GpuData::make(size);
        CppData::Pointer cppData2 = CppData::make(size);
        gpuData->copy(*cppData);
        cppData2->copy(*gpuData);
        CHECK(*cppData == *cppData2);
    }

    TEST_FIXTURE(DataTestFixture, CppData_copy)
    {
        CHECK_EQUAL(size, cppData->getSize());
        CppData::Pointer cppData2 = CppData::make(size);
        cppData2->copy(*cppData);
        CHECK(*cppData == *cppData2);
    }
}
