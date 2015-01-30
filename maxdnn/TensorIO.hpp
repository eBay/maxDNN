/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_TensorIO_h
#define maxdnn_TensorIO_h

#include "maxdnn/File.hpp"
#include "maxdnn/Tensor.hpp"
#include "maxdnn/MappedBlockData.hpp"
#include <string>

namespace maxdnn
{
    bool write(const Shape& shape, File &file);

    template<typename Scalar>
    inline
    bool write(const Tensor<Scalar> &T, File &file)
    {
        const Shape &shape = T.getShape();
        return 
            write(shape, file) &&
            file.write(&T(), shape.getSize<Scalar>());
    }
    
    template<typename Scalar>
    inline
    bool write(const Tensor<Scalar> &T, const std::string &fileName)
    {
        File file(fileName, File::Write|File::Create);
        return write(T, file);
    }

    template<typename Scalar>
    inline void read(Tensor<Scalar> &T, const std::string &fileName)
    {
        T.create(MappedBlockData::read(fileName));
    }
}

#endif
