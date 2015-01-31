/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/FileSystem.hpp"
#include "maxdnn/FileName.hpp"
#include "maxdnn/test.hpp"
#include "UnitTest++.h"
using namespace maxdnn;
using namespace std;
namespace fs=maxdnn::FileSystem;

struct FileSystemTestFixture
{
    FileSystemTestFixture()
    {
        FileName testDir(getTestDataDirectory());
        if (!testDir.isEmpty()) {
            path = testDir / "fs-test";
            path1 = testDir / "fs-test/foo";
            path2 = testDir / "fs-test/foo/bar";
            path3 = testDir / "fs-test/foo/baz";
            fileName = testDir / "fs-test/foo/baz/file";
        }
    }

    ~FileSystemTestFixture()
    {
    }
    
    FileName path;
    FileName path1;
    FileName path2;
    FileName path3;
    FileName fileName;
};
    
SUITE(FileSystem)
{
    TEST_FIXTURE(FileSystemTestFixture, Directories)
    {
        if (!path.isEmpty()) {
            fs::deltree(path.getString());
            CHECK(!fs::exists(path.getString()));
            CHECK(fs::mkdir(path.getString()));
            CHECK(fs::mkdir(path1.getString()));
            CHECK(fs::mkdir(path2.getString()));
            CHECK(fs::mkdir(path3.getString()));
            CHECK(fs::exists(path3.getString()));
            CHECK(fs::isDirectory(path.getString()));
            CHECK(fs::touch(fileName.getString()));
            CHECK(fs::exists(fileName.getString()));
            CHECK(fs::isRegularFile(fileName.getString()));
            CHECK(fs::deltree(path.getString()));
            CHECK(!fs::exists(path.getString()));
        }
    }

    TEST_FIXTURE(FileSystemTestFixture, mkpath)
    {
        if (!path.isEmpty()) {
            fs::deltree(path3.getString());
            CHECK(!fs::exists(path3.getString()));
            CHECK(fs::mkpath(path3.getString()));
            CHECK(fs::isDirectory(path3.getString()));
            fs::deltree(path3.getString());
        }
    }
}
