/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include <nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>
#include "maxdnn/profile.hpp"
#include <cstdio>
#include <cstdarg>

CudaProfileRange::CudaProfileRange(const char *format, ...) {
    char name[256];
    va_list ap;

    va_start(ap, format);
    vsnprintf(name, sizeof(name), format, ap);
    va_end(ap);

    name[sizeof(name)-1]=0;

    nvtxRangePush(name);
}

CudaProfileRange::~CudaProfileRange() {
    nvtxRangePop();
}

void profileStart() {
    cudaProfilerStart();
}

void profileStop() {
    cudaProfilerStop();
}

void profileEnd() {
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void profileNameOsThread(unsigned threadId, const char *format, ...) {
    char name[256];
    va_list ap;

    va_start(ap, format);
    vsnprintf(name, sizeof(name), format, ap);
    va_end(ap);

    name[sizeof(name)-1]=0;

    nvtxNameOsThread(threadId, name);
}
