/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "UnitTest++.h"
#include "maxdnn/Tensor.hpp"
#include "maxdnn/TensorIO.hpp"
#include "maxdnn/File.hpp"
#include "maxdnn/FileSystem.hpp"
#include "maxdnn/MappedBlockData.hpp"
#include "maxdnn/CppData.hpp"
namespace fs = maxdnn::FileSystem;
using namespace maxdnn;
using namespace std;
    
struct TensorIOTestFixture
{
    TensorIOTestFixture()
        : T(5, 6, 7, 8)
    {
        fileName = "/tmp/maxdnn-TensorIO-test-file";
        
        const Shape &s = T.getShape();
        
        for (int g = 0; g < s.getNumImages(); ++g) {
            for (int h = 0; h < s.getNumChannels(); ++h) {
                for (int i = 0; i < s.getNumRows(); ++i) {
                    for (int j = 0; j < s.getNumColumns(); ++j) {
                        T(g, h, i, j) = g*h*i*j;
                    }
                }
            }
        }
        
    }
    
    ~TensorIOTestFixture()
    {
    }

    Tensor<Float> T;
    string fileName;
};

    
SUITE(Tensor)
{
    TEST_FIXTURE(TensorIOTestFixture, write)
    {
        fs::remove(fileName);

        // Write the tensor to a file.
        CHECK(write(T, fileName));

        // Load the tensor from the file.
        Tensor<Float> T2;
        read(T2, fileName);

        // Check that the loaded tensor value equals the original.
        CHECK(T==T2);

        CHECK(fs::remove(fileName));
    }
}
