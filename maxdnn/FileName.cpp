/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/FileName.hpp"
#include <iostream>
using namespace std;


namespace
{
    string removeTrailingSeparator(const string &fileName);
}


namespace maxdnn
{
    string FileName::getParent() const
    {
        string parent = removeTrailingSeparator(_fileName);
        size_t i = parent.rfind(PathSeparator);
        if (i != string::npos) {
            parent = parent.substr(0, i);
        } else {
            parent = "";
        }
        return parent;
    }

    string FileName::getBaseName() const
    {
        string basename = removeTrailingSeparator(_fileName);
        size_t i = basename.rfind(PathSeparator);
        if (i != string::npos) {
            basename = basename.substr(i+1);
        }
        return basename;
     }

    std::ostream &operator<<(std::ostream &s, const FileName &fileName)
    {
        return s << fileName;
    }
}

namespace
{
    using namespace maxdnn;
    
    string removeTrailingSeparator(const string &fileName)
    {
        string trimmed = fileName;
        if (!trimmed.empty() && trimmed[trimmed.size()-1] == FileName::PathSeparator) {
            trimmed.resize(trimmed.size()-1);
        }
        return trimmed;
    }
}
