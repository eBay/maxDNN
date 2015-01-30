/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "UnitTest++.h"
#include "maxdnn/cuDNNContext.hpp"
using namespace maxdnn;
using namespace std;
    
SUITE(cuDNNContext)
{
    TEST(create)
    {
        cuDNNContext ctx;
        
        ctx.create();
    }
}
