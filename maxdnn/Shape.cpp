/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/Shape.hpp"
#include <iostream>
#include <sstream>
using namespace std;


namespace maxdnn
{
    string Shape::getDescription() const
    {
        ostringstream s;
        s << *this;
        return s.str();
    }
    
    ostream & operator<<(ostream &s, const Shape &shape)
    {
        return s << "{ " << shape.K << ", " << shape.L << ", " << shape.M << ", " << shape.N;
    }
    
}

