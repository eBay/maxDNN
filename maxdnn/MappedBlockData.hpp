/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_MappedBlockData_h
#define maxdnn_MappedBlockData_h

#include "maxdnn/BlockData.hpp"
#include "maxdnn/Shape.hpp"
#include "maxdnn/MemoryMap.hpp"
#include "maxdnn/File.hpp"

namespace maxdnn
{
    /// Memory mapped blockd data.
    class MappedBlockData : public BlockData
    {
    public:

        typedef std::tr1::shared_ptr<MappedBlockData> Pointer;
        
        Data::Pointer clone(size_t size) const
        {
            // TODO
            return Pointer();
        }
        
        static Pointer make();

        static Pointer read(const std::string &fileName);

        /// Memory-map the block data from the given file at the given
        /// offset. Offset must be a multiple of the page size.
        bool map(File &file, size_t offset=0);

        /// Unmap the block data.
        bool unmap();

        ~MappedBlockData();

        void *getData();

        const void *getData() const;

        size_t getSize() const;

    protected:

        MappedBlockData();
        MappedBlockData(const std::string &fileName);

        typedef std::tr1::shared_ptr<File> FilePtr;

        MemoryMap _map;
        void *_data;
        FilePtr _file;
    };
}


#endif
