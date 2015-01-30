/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/MappedBlockData.hpp"
#include "maxdnn/File.hpp"

namespace maxdnn
{
    MappedBlockData::Pointer MappedBlockData::make()
    {
        return Pointer(new MappedBlockData());
    }

    MappedBlockData::Pointer MappedBlockData::read(const std::string &fileName)
    {
        return Pointer(new MappedBlockData(fileName));
    }
    
    bool MappedBlockData::map(File &file, size_t offset)
    {
        file.seek(offset, File::Current);
        if (!file.read(&_shape, sizeof(_shape))) {
            return false;
        }
        if (_map.map(file.getDescriptor(),
                     // TODO: allow for blocks of arbitray scalar
                     // type. For now, just float.
                     _shape.getSize<float>()+sizeof(_shape),
                     MemoryMap::Read,
                     MemoryMap::Private,
                     offset)) {
            _data = _map.getMemory() + sizeof(_shape);
            return true;
        }
        return false;
    }

    bool MappedBlockData::unmap()
    {
        _shape = Shape(0, 0, 0, 0);
        return _map.unmap();
    }

    MappedBlockData::~MappedBlockData()
    {
    }

    void *MappedBlockData::getData()
    {
        return _data;
    }

    const void *MappedBlockData::getData() const
    {
        return _data;
    }

    size_t MappedBlockData::getSize() const
    {
        return _map.getLength();
    }

    MappedBlockData::MappedBlockData()
    {
        _data = 0;
    }

    MappedBlockData::MappedBlockData(const std::string &fileName)
        : _data(0),
          _file(new File(fileName, File::Read))
    {
        if (_file->isOk()) {
            map(*_file);
        }
    }
}
