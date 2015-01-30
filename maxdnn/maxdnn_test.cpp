/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/maxdnn_test.hpp"
#include "maxdnn/FileName.hpp"
#include <stdlib.h>
using namespace maxdnn;
using namespace std;


namespace
{
    const char *maxdnn_test_data_env = "maxdnn_test_data";
    const char *maxdnn_test_device = "maxdnn_test_device";
    const char *maxdnn_test_generate_reference_output = "maxdnn_test_generate_reference_output";
    const char *maxdnn_test_no_test = "maxdnn_test_no_test";
    const char *maxdnn_test_conv_iters = "maxdnn_test_conv_iters";
    const char *maxdnn_test_cooldown = "maxdnn_test_cooldown";
    const int DefaultDevice = 0;
    const int DefaultConvIters = 10;
    const int DefaultCooldown = 0;

    int getEnvFlag(const char* varname, int defaultValue=0)
    {
        int flag = defaultValue;
        char *flag_str = getenv(varname);
        if (flag_str) {
            flag = atoi(flag_str);
        }
        return flag;
    }
}

string getTestDataDirectory() throw(maxdnn::EnvironmentException)
{
    char *path = getenv(maxdnn_test_data_env);
    if (path == 0) {
        return "";
    }
    return string(path);
}

string getTestFile(const string &name) throw(maxdnn::EnvironmentException)
{
    return FileName(getTestDataDirectory(), name).getString();
}

bool generateReferenceOutput()
{
    return getEnvFlag(maxdnn_test_generate_reference_output) != 0;
}

bool noTest()
{
    return getEnvFlag(maxdnn_test_no_test) != 0 || getTestDataDirectory().empty();
}

int getConvIters()
{
    return getEnvFlag(maxdnn_test_conv_iters, DefaultConvIters);
}

int getCooldown()
{
    return getEnvFlag(maxdnn_test_cooldown, DefaultCooldown);
}

int getTestDevice()
{
    return getEnvFlag(maxdnn_test_device, DefaultDevice);
}
