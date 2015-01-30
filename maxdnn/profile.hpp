/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef MAXDNN_PROFILE_H
#define MAXDNN_PROFILE_H

#ifndef MAXDNN_PROFILE_DISABLE
#define MAXDNN_PROFILE_NAME_OS_THREAD(THREADID, ...) \
    profileNameOsThread(THREADID, __VA_ARGS__)
#define MAXDNN_SCOPE(...) CudaProfileRange cuda_convnet_profile_scope ## __COUNTER__ (__VA_ARGS__)
#define MAXDNN_PROFILE_START profileStart()
#define MAXDNN_PROFILE_STOP profileStop()
#define MAXDNN_PROFILE_END profileEnd();
#define MAXDNN_PROFILE_BEGIN_RANGE(...) profileBeginRange(__VA_ARGS__)
#define MAXDNN_PROFILE_END_RANGE profileEndRange()
#define MAXDNN_CPU_PROFILE_WRITE_REPORT(STREAM) cpuProfileWriteReport(STREAM)
#define MAXDNN_CPU_SCOPE(...) CpuProfileRange cuda_convnet_cpu_profile_scope ## __COUNTER__ (__VA_ARGS__)
#else
#define MAXDNN_PROFILE_NAME_OS_THREAD(THREADID, ...)
#define MAXDNN_SCOPE(...)
#define MAXDNN_PROFILE_START
#define MAXDNN_PROFILE_STOP
#define MAXDNN_PROFILE_END
#define MAXDNN_CPU_SCOPE(...)
#define MAXDNN_PROFILE_BEGIN_RANGE(...)
#define MAXDNN_PROFILE_END_RANGE
#define MAXDNN_CPU_PROFILE_WRITE_REPORT(STREAM) (STREAM << "Profiling is disabled.")
#endif

class CudaProfileRange
{
public:
    CudaProfileRange(const char *format, ...);
    ~CudaProfileRange();
};

void profileNameOsThread(unsigned threadId, const char *format, ...);
void profileStart();
void profileStop();
void profileEnd();
void profileBeginRange(const char *format, ...);
void profileEndRange();

#endif
