/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_test_hpp
#define maxdnn_test_hpp

#include "UnitTest++.h"
#include "maxdnn/Tensor.hpp"
#include "maxdnn/Exception.hpp"
#include <cmath>

template< typename Expected, typename Actual, typename Tolerance >
inline
bool AreRelClose(Expected const& expected, Actual const& actual, Tolerance const& tolerance)
{
    if (fabs(expected) < Tolerance(0.1)) {
        return (actual >= (expected - tolerance)) && (actual <= (expected + tolerance));
    } else {
        const Expected d = fabs(expected)*tolerance;
        return (actual >= (expected-d)) && (actual <= (expected+d));
    }
}

template< typename Expected, typename Actual, typename Tolerance >
inline
void CheckRelClose(UnitTest::TestResults& results, Expected const& expected, Actual const& actual, Tolerance const& tolerance, UnitTest::TestDetails const& details)
{
    if (!AreRelClose(expected, actual, tolerance))
    { 
        UnitTest::MemoryOutStream stream;
        stream << "Expected " << expected << " with relative tolerance " << tolerance << " but was " << actual;
        results.OnTestFailure(details, stream.GetText());
    }
}

template< typename Scalar>
inline
void CheckRelClose(UnitTest::TestResults& results, 
                   const maxdnn::Tensor<Scalar>& expected,
                   int eg, int eh, int ei, int ej,
                   const maxdnn::Tensor<Scalar>& actual,
                   Scalar tolerance,
                   UnitTest::TestDetails const& details)
{
    if (!AreRelClose(expected(eg,eh,ei,ej), actual(eg,eh,ei,ej), tolerance))
    { 
        UnitTest::MemoryOutStream stream;
        stream << "Expected (" << eg << "," << eh << "," << ei << "," << ej << ")=" << expected(eg,eh,ei,ej)
               << " with relative tolerance " << tolerance 
               << " but was " << actual(eg,eh,ei,ej);
        results.OnTestFailure(details, stream.GetText());
    }
}

#define CHECK_REL_CLOSE(expected, actual, tolerance) \
    do \
    { \
        try { \
            CheckRelClose(*UnitTest::CurrentTest::Results(), expected, actual, tolerance, UnitTest::TestDetails(*UnitTest::CurrentTest::Details(), __LINE__)); \
        } \
        catch (...) { \
            UnitTest::CurrentTest::Results()->OnTestFailure(UnitTest::TestDetails(*UnitTest::CurrentTest::Details(), __LINE__), \
                    "Unhandled exception in CHECK_REL_CLOSE(" #expected ", " #actual ")"); \
        } \
    } while (0)

void checkRelClose(const maxdnn::Tensor<maxdnn::Float> &T,
                   const maxdnn::Tensor<maxdnn::Float> &T2,
                   maxdnn::Float tolerance);

std::string getTestDataDirectory() throw(maxdnn::EnvironmentException);
std::string getTestFile(const std::string &name) throw(maxdnn::EnvironmentException);

class TestMain
{
public:

    int main(int argc, const char** argv);

    virtual void startup() 
    {
    }
    
    virtual void teardown()
    {
    }
};


#endif
