/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/maxdnn_test.hpp"
#include "maxdnn/profile.hpp"
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

class MyTestMain : public TestMain
{
    void startup()
    {
        if (generateReferenceOutput()) {
            cout << "generate reference output: " << generateReferenceOutput() << endl;
        }
        
        if (noTest()) {
            cout << "no test: " << noTest() << endl;
        }

//        if (getConvIters() != DefaultConvIters) {
            cout << "convolution iterations: " << getConvIters() << endl;
//        }

        int device = getTestDevice();;

        cout << "Using GPU device " << device << endl;
        
        MAXDNN_PROFILE_START;
        cudaSetDevice(device);
    }
    
    void teardown()
    {
        // Ensure that all kernels have finished.
        cudaDeviceSynchronize();

        MAXDNN_PROFILE_STOP;

        // Ensure that the device is reset so that any nvprof profiling is flushed.
        cudaDeviceReset();
    }
};

int main( int argc, const char** argv )
{
    MyTestMain myMain;
    
    return myMain.main(argc, argv);
}

