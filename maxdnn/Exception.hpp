/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Exception_hpp
#define maxdnn_Exception_hpp

#include <exception>
#include <string>
#include <sstream>

#define MAXDNN_ASSERT(EXPR, EXCEPTION) \
    if (!(EXPR)) throw EXCEPTION;

#define MAXDNN_ASSERT_SIZE_MATCH(SIZE1, SIZE2, VARIABLE)                \
    {                                                                \
    if (!((SIZE1)==(SIZE2))) {                                       \
        throw SizeMismatchException(VARIABLE, (SIZE1), (SIZE2));     \
    }                                                                \
    }                                                                \

namespace maxdnn
{
    class Exception : public std::exception
    {
    public:

        virtual ~Exception() throw();

        const char *what() const throw();
        
        virtual std::string getDescription() const = 0;

    protected:

        template<typename Type>
        static std::string toString(const Type &value)
        {
            std::ostringstream s;
            s << value;
            return s.str();
        }

        mutable std::string _what;
    };

    class BadArgException : public Exception
    {
    public:

        template<typename Value, typename Expect>
        BadArgException(const std::string &argName,
                        const Value &value,
                        const Expect &expected)
            : _argName(argName),
              _value(toString(value)),
              _expected(toString(expected))
        {
        }
        
        ~BadArgException() throw();

        const std::string &getArgName() const;

        const std::string &getValue() const;
        
        const std::string &getExpected() const;
        
        std::string getDescription() const;

    private:

        std::string _argName;
        std::string _value;
        std::string _expected;
    };
    
    class AllocException : public Exception
    {
    public:

        AllocException(size_t size, const std::string &memoryType);
        
        virtual ~AllocException() throw();

        const std::string &getMemoryType() const;

        size_t getSize() const;

        std::string getDescription() const;

    private:
        std::string _memoryType;
        size_t _size;
    };

    class BoundsException : public Exception
    {
    public:

        BoundsException(size_t value,
                        size_t bound,
                        const std::string &variable);
        
        virtual ~BoundsException() throw();

        size_t getValue() const;
        
        size_t getBound() const;
        
        const std::string &getVariable() const;
        
        std::string getDescription() const;

    private:
        size_t _value;
        size_t _bound;
        std::string _variable;
        size_t _size;
    };

    class SizeMismatchException : public Exception
    {
    public:

        template<class SizeType>
        SizeMismatchException(const std::string &variable,
                              const SizeType &size1,
                              const SizeType &size2)
            : _variable(variable),
              _size1(toString(size1)),
              _size2(toString(size2))
        {
        }
        
        ~SizeMismatchException() throw();

        const std::string &getVariable() const;

        const std::string &getSize1() const;
        
        const std::string &getSize2() const;
        
        std::string getDescription() const;

    private:

        std::string _variable;
        std::string _size1;
        std::string _size2;
    };

    class EnvironmentException : public Exception
    {
    public:

        EnvironmentException(const std::string &variable);
                
        ~EnvironmentException() throw();

        const std::string &getVariable() const;

        std::string getDescription() const;

    private:

        std::string _variable;
    };

    
}

#endif 
