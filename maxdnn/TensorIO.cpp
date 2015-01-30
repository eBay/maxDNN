/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/Shape.hpp"
#include "maxdnn/File.hpp"

namespace maxdnn
{
    bool write(const Shape& shape, File &file)
    {
        return file.write(&shape, sizeof(shape));
    }
}
