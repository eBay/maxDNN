/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/FileSystem.hpp"
#include "UnitTest++.h"
using namespace maxdnn;
using namespace std;
namespace fs=maxdnn::FileSystem;

struct FileSystemTestFixture
{
    FileSystemTestFixture()
    {
        path = "/tmp/fs-test";
        path1 = "/tmp/fs-test/foo";
        path2 = "/tmp/fs-test/foo/bar";
        path3 = "/tmp/fs-test/foo/baz";
        fileName = "/tmp/fs-test/foo/baz/file";
    }

    ~FileSystemTestFixture()
    {
    }
    
    string path;
    string path1;
    string path2;
    string path3;
    string fileName;
};
    
SUITE(FileSystem)
{
    TEST_FIXTURE(FileSystemTestFixture, Directories)
    {
        fs::deltree(path);
        CHECK(!fs::exists(path));
        CHECK(fs::mkdir(path));
        CHECK(fs::mkdir(path1));
        CHECK(fs::mkdir(path2));
        CHECK(fs::mkdir(path3));
        CHECK(fs::exists(path3));
        CHECK(fs::isDirectory(path));
        CHECK(fs::touch(fileName));
        CHECK(fs::exists(fileName));
        CHECK(fs::isRegularFile(fileName));
        CHECK(fs::deltree(path));
        CHECK(!fs::exists(path));
    }

    TEST_FIXTURE(FileSystemTestFixture, mkpath)
    {
        fs::deltree(path3);
        CHECK(!fs::exists(path3));
        CHECK(fs::mkpath(path3));
        CHECK(fs::isDirectory(path3));
        fs::deltree(path3);
    }
}
