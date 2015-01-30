/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/test.hpp"
#include "TestReporterStdout.h"
#include <string.h>
#include <stdlib.h>
using namespace UnitTest;
using namespace maxdnn;
using namespace std;


/// Predicate that is true for tests with matching name,
/// or all tests if no names were given.
class Predicate
{
public:

    Predicate(const char **tests, int nTests)
        : _tests(tests),
          _nTests(nTests)
    {
    }

    bool operator()(Test *test) const
    {
        bool match = (_nTests == 0);
        
        for (int i = 0; !match && i < _nTests; ++i) {
            if (!strcmp(test->m_details.testName, _tests[i])) {
                match = true;
            }
        }
        return match;
    }

private:

    const char **_tests;
    int _nTests;
};

int TestMain::main(int argc, const char** argv)
{
    startup();

    const char *suiteName = 0;
    int arg = 1;

    // If the first arg is "suite", then the second arg must be a
    // suite name.
    if (argc >=3 && strcmp( "suite", argv[arg] ) == 0) {
        // Select all tests in the suite.
        suiteName = argv[++arg];
        ++arg;
    } 

    // Construct predicate that matches any tests given on command line.
    Predicate pred(argv + arg, argc - arg);

    // Run tests that match any given suite name and any given tests
    // names.
    TestReporterStdout reporter;
    TestRunner runner( reporter );
    int r = runner.RunTestsIf(Test::GetTestList(), suiteName, pred, 0);

    teardown();
        
    return r;
}

void checkRelClose(const Tensor<Float> &expected,
                   const Tensor<Float> &actual,
                   Float tolerance)
{
    for (int g = 0; g < expected.getShape().K; ++g) {
        for (int h = 0; h < expected.getShape().L; ++h) {
            for (int i = 0; i < expected.getShape().M; ++i) {
                for (int j = 0; j < expected.getShape().N; ++j) {
                    CheckRelClose(*UnitTest::CurrentTest::Results(),
                                  expected, g, h, i, j, actual, tolerance, 
                                  UnitTest::TestDetails(*UnitTest::CurrentTest::Details(), __LINE__));
                }
            }
        }
    }
}
