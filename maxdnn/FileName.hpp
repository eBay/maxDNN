/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_FileName_h
#define maxdnn_FileName_h

#include <string>
#include <iosfwd>

namespace maxdnn
{
    class FileName
    {
    public:

        enum { PathSeparator = '/' };
            
        FileName() {}

        FileName(const std::string &fileName) : _fileName(fileName) {}

        FileName(const std::string &path, const std::string &fileName)
            : _fileName(path)
        {
            *this += fileName;
        }

        FileName(const std::string &path, 
                 const std::string &subpath,
                 const std::string &fileName)
            : _fileName(path)
        {
            *this += subpath;
            *this += fileName;
        }

        FileName(const std::string &path, 
                 const std::string &subpath,
                 const std::string &subsubpath,
                 const std::string &fileName)
            : _fileName(path)
        {
            *this += subpath;
            *this += subsubpath;
            *this += fileName;
        }

        FileName &operator+=(const std::string &part)
        {
            _fileName += PathSeparator;
            _fileName += part;
            return *this;
        }
        
        FileName operator+(const std::string &part) const
        {
            return FileName(*this) += part;
        }
        
        /// Alternate syntax for file name concatenation.
        FileName operator/(const std::string &part) const
        {
            return *this + part;
        }
        
        /// Alternate syntax for file name concatenation.
        FileName &operator/=(const std::string &part)
        {
            return *this += part;
        }

        const std::string &getString() const 
        {
            return _fileName;
        }
        
        const char *getCstring() const 
        {
            return _fileName.c_str();
        }

        std::string getParent() const;

        std::string getBaseName() const;
        
        bool isEmpty() const 
        {
            return _fileName.empty();
        }

        bool operator==(const FileName &other) const 
        {
            return _fileName == other._fileName;
        }

    private:

        std::string _fileName;
    };

    std::ostream &operator<<(std::ostream &s, const FileName &fileName);
}

#endif
