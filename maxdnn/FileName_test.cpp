/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/FileName.hpp"
#include "UnitTest++.h"
using namespace maxdnn;
using namespace std;


SUITE(FileName)
{
    TEST(Constructors)
    {
        CHECK_EQUAL("foo", FileName("foo").getString());
        CHECK_EQUAL("foo/bar", FileName("foo", "bar").getString());
        CHECK_EQUAL("foo/bar/baz", FileName("foo", "bar", "baz").getString());
        CHECK_EQUAL("foo/bar/baz/quux", FileName("foo", "bar", "baz", "quux").getString());
    }

    TEST(Operators)
    {
        CHECK_EQUAL(FileName("foo/bar"), FileName("foo") / "bar");
        CHECK_EQUAL(FileName("foo/bar"), FileName("foo") + "bar");
        CHECK_EQUAL(FileName("foo/bar"), FileName("foo") += "bar");
        CHECK_EQUAL(FileName("foo/bar"), FileName("foo") /= "bar");
        CHECK_EQUAL(FileName("foo/bar/baz"), FileName("foo") / "bar" / "baz");
    }

    TEST(Parts)
    {
        CHECK_EQUAL("foo", FileName("foo", "bar").getParent());
        CHECK_EQUAL("foo", FileName("foo", "bar/").getParent());
        CHECK_EQUAL("", FileName("foo").getParent());
        CHECK_EQUAL("bar", FileName("foo", "bar").getBaseName());
        CHECK_EQUAL("bar", FileName("foo", "bar/").getBaseName());
        CHECK_EQUAL("foo", FileName("foo").getBaseName());
    }

    TEST(Accessors)
    {
        CHECK_EQUAL("foo/bar", FileName("foo", "bar").getString());
        CHECK_EQUAL("foo/bar", FileName("foo", "bar").getCstring());
    }
}


    

