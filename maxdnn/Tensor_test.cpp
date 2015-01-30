/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/test.hpp"
#include "maxdnn/Tensor.hpp"
#include "maxdnn/TensorRef.hpp"
#include "maxdnn/GpuData.hpp"
#include "maxdnn/File.hpp"
#include "maxdnn/FileSystem.hpp"
#include "maxdnn/convolution.hpp"
namespace fs = maxdnn::FileSystem;
using namespace maxdnn;
using namespace std;
    
struct TensorTestFixture
{
    TensorTestFixture()
        : padding(5),
          shape(5, 6, 321, 272,
                272, (321+padding)*272, (6+padding)*(321+padding)*272),
          T(shape),
          T2(shape)
    {
        fileName = "/tmp/Tenstor-test-file";
        
        for (int g = 0; g < shape.K; ++g) {
            for (int h = 0; h < shape.L; ++h) {
                for (int i = 0; i < shape.M; ++i) {
                    for (int j = 0; j < shape.N; ++j) {
                        T(g, h, i, j) = g*h*i*j;
                        T2(g, h, i, j) = 1./(max(1, g*h*i*j));
                    }
                }
            }
        }
    }
    
    ~TensorTestFixture()
    {
    }

    int padding;
    Shape shape;
    Tensor<Float> T;
    Tensor<Float> T2;
    string fileName;
};


SUITE(Tensor)
{
    TEST_FIXTURE(TensorTestFixture, size)
    {
        CHECK_EQUAL(5u*6u*321u*272u, T.getShape().getNumElements());

        fs::remove(fileName);
        
        File file(fileName, File::Write|File::Create);
        CHECK(file.write(&T(), T.getShape().getSize<Float>()));
        CHECK(file.close());
        CHECK_EQUAL(off_t(T.getShape().getSize<Float>()), fs::getFileSize(fileName));
    }

    TEST_FIXTURE(TensorTestFixture, accesors)
    {
        CHECK_EQUAL(Float(2*3*4*5), T(2, 3, 4, 5));
        CHECK_EQUAL(Float(4*5*6*7), T(4, 5, 6, 7));
        CHECK_EQUAL(Float(0), T(4, 5, 6));
        CHECK_EQUAL(Float(0), T(4, 5));
        CHECK_EQUAL(Float(0), T(4));
    }

    TEST_FIXTURE(TensorTestFixture, window)
    {
        Tensor<Float>::Ref Tw = T.window(1, 2, 3, 4,
                                         3, 4, 200, 250);
        
        CHECK_EQUAL(3, Tw.getShape().K);
        CHECK_EQUAL(4, Tw.getShape().L);
        CHECK_EQUAL(200, Tw.getShape().M);
        CHECK_EQUAL(250, Tw.getShape().N);
        CHECK_EQUAL(T.getShape().strideK, Tw.getShape().strideK);
        CHECK_EQUAL(T.getShape().strideL, Tw.getShape().strideL);
        CHECK_EQUAL(T.getShape().strideM, Tw.getShape().strideM);
        CHECK_EQUAL(T(1, 2, 3, 4), Tw(0, 0, 0, 0));
        CHECK_EQUAL(T(2, 3, 4, 5), Tw(1, 1, 1, 1));
        CHECK_EQUAL(T(3, 4, 5, 6), Tw(2, 2, 2, 2));
    }

    TEST_FIXTURE(TensorTestFixture, tensorRefWindow)
    {
        TensorRef<Float> T_ref = T.getRef();
        
        CHECK_EQUAL(T.getShape(), T_ref.getShape());

        TensorRef<Float> Tw = T_ref.window(1, 2, 3, 4,
                                           3, 4, 200, 250);
        CHECK_EQUAL(3, Tw.getShape().K);
        CHECK_EQUAL(4, Tw.getShape().L);
        CHECK_EQUAL(200, Tw.getShape().M);
        CHECK_EQUAL(250, Tw.getShape().N);
        CHECK_EQUAL(T.getShape().strideK, Tw.getShape().strideK);
        CHECK_EQUAL(T.getShape().strideL, Tw.getShape().strideL);
        CHECK_EQUAL(T.getShape().strideM, Tw.getShape().strideM);
        CHECK_EQUAL(T(1, 2, 3, 4), Tw(0, 0, 0, 0));
        CHECK_EQUAL(T(2, 3, 4, 5), Tw(1, 1, 1, 1));
        CHECK_EQUAL(T(3, 4, 5, 6), Tw(2, 2, 2, 2));
    }

    TEST_FIXTURE(TensorTestFixture, cpu_gpu_copy)
    {
        Tensor<Float> T_gpu(T, GpuData::prototype());
        Tensor<Float> T_cpu(T_gpu);
        checkRelClose(T, T_cpu, 0.f);
    }
}
