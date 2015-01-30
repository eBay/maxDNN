/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "UnitTest++.h"
#include "maxdnn/Random.hpp"
using namespace maxdnn;
    
SUITE(Random)
{
    TEST(Uniform)
    {
        unsigned seed = 12345;

        Random randm(seed);
        
        CHECK_CLOSE(0.48279, randm.uniform(), 1e-5);
        CHECK_CLOSE(0.708267, randm.uniform(), 1e-5);
        CHECK_CLOSE(0.639569, randm.uniform(), 1e-5);
        CHECK_CLOSE(0.987021, randm.uniform(), 1e-5);
    }

    TEST(Uniform_range)
    {
        unsigned seed = 12345;

        Random randm(seed);
        double min = 100;
        double max = 1000;
        
        CHECK_CLOSE(min + (max-min)*0.48279, randm.uniform(min, max), 1e-2);
        CHECK_CLOSE(min + (max-min)*0.708267, randm.uniform(min, max), 1e-2);
        CHECK_CLOSE(min + (max-min)*0.639569, randm.uniform(min, max), 1e-2);
        CHECK_CLOSE(min + (max-min)*0.987021, randm.uniform(min, max), 1e-2);
    }
}
