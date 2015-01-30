/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/Exception.hpp"
#include <sstream>
using namespace std;

namespace maxdnn
{
    Exception::~Exception() throw()
    {
    }
    
    const char *Exception::what() const throw()
    {
        _what = getDescription();
        return _what.c_str();
    }

    BadArgException::~BadArgException() throw()
    {
    }
    
    const string &BadArgException::getArgName() const
    {
        return _argName;
    }
    
    const string &BadArgException::getValue() const
    {
        return _value;
    }
    
    const string &BadArgException::getExpected() const
    {
        return _expected;
    }
    
    string BadArgException::getDescription() const
    {
        ostringstream s;
        s << "Bad value " << _value << " for argument " << _argName << ", expected " << _expected;
        return s.str();
    }

    AllocException::AllocException(size_t size, const string &memoryType)
        : _memoryType(memoryType),
          _size(size)
    
    {
    }
        
    AllocException::~AllocException() throw()
    {
    }
    
    const string &AllocException::getMemoryType() const
    { 
        return _memoryType;
    }

    size_t AllocException::getSize() const
    {
        return _size;
    }

    string AllocException::getDescription() const
    {
        ostringstream s;
        s << "Failed to allocate " << _size << " bytes from " << _memoryType;
        return s.str();
    }

    BoundsException::BoundsException(size_t value,
                                     size_t bound,
                                     const string &variable)
        : _value(value),
          _bound(bound),
          _variable(variable)
    {
    }
        
    BoundsException::~BoundsException() throw()
    {
    }

    size_t BoundsException::getValue() const
    {
        return _value;
    }
        
    size_t BoundsException::getBound() const
    {
        return _bound;
    }
        
    const string &BoundsException::getVariable() const
    {
        return _variable;
    }
    
        
    string BoundsException::getDescription() const
    {
        ostringstream s;
        s << "Bounds exception for variable " << _variable
          << " requested value " << _value
          << " bound is " << _bound;
        return s.str();
    }

    SizeMismatchException::~SizeMismatchException() throw()
    {
    }
    
    const string &SizeMismatchException::getVariable() const
    {
        return _variable;
    }
    

    const string &SizeMismatchException::getSize1() const
    {
        return _size1;
    }
    
        
    const string &SizeMismatchException::getSize2() const
    {
        return _size2;
    }
    
        
    string SizeMismatchException::getDescription() const
    {
        ostringstream s;
        s << "size mismatch for variable " << _variable
          << " size1 " << _size1
          << " size2 " << _size2;
        return s.str();
    }

    EnvironmentException::EnvironmentException(const string &variable)
        : _variable(variable)
    {
    }
        
    EnvironmentException::~EnvironmentException() throw()
    {
    }

    const string &EnvironmentException::getVariable() const
    {
        return _variable;
    }
    

    string EnvironmentException::getDescription() const
    {
        ostringstream s;
        s << "undefined environment variable " << _variable;
        return s.str();
    }
}
