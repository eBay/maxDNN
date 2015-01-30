/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "UnitTest++.h"
#include "maxdnn/Shape.hpp"
using namespace maxdnn;
    
struct ShapeTestFixture
{
    ShapeTestFixture()
        : shape(5, 6, 7, 8),
          shape2(5, 6, 7, 8, 8, 56, 336)
    {
    }
    
    ~ShapeTestFixture()
    {
    }

    Shape shape;
    Shape shape2;
};

    
SUITE(Shape)
{
    TEST_FIXTURE(ShapeTestFixture, size)
    {
        CHECK_EQUAL(1680u, shape.getSize<unsigned char>());
    }

    TEST_FIXTURE(ShapeTestFixture, accesors)
    {
        CHECK_EQUAL(5, shape.getNumImages());
        CHECK_EQUAL(6, shape.getNumChannels());
        CHECK_EQUAL(7, shape.getNumRows());
        CHECK_EQUAL(8, shape.getNumColumns());
    }

    TEST_FIXTURE(ShapeTestFixture, indexing)
    {
        CHECK_EQUAL(336, shape(1));
        CHECK_EQUAL(336+2*56, shape(1, 2));
        CHECK_EQUAL(336+2*56+3*8, shape(1, 2, 3));
        CHECK_EQUAL(336+2*56+3*8+4, shape(1, 2, 3, 4));
    }

    TEST_FIXTURE(ShapeTestFixture, strides)
    {
        CHECK_EQUAL(shape, shape2);
        CHECK_EQUAL(shape(2, 3, 4, 5), shape2(2, 3, 4, 5));
    }
}
