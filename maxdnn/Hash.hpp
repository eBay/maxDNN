/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Hash_h
#define maxdnn_Hash_h

#include <stdint.h>
#include <tr1/unordered_map>

namespace maxdnn
{
    // FNV hash, as described at:
    // http://www.eternallyconfuzzled.com/tuts/algorithms/jsw_tut_hashing.aspx
    template<class Record>
    struct Hash
    {
        static void _fnv_hash_update(unsigned &h, uint8_t byte)
        {
            h = (h * 16777619) ^ byte;
        }

        static void _fnv_hash_update(unsigned &h, const void *data, unsigned len)
        {
            const uint8_t *a = static_cast<const uint8_t *>(data);
            for (unsigned i=0; i<len; ++i) {
                _fnv_hash_update(h, a[i]);
            }
        }
            
        size_t operator()(const Record &record) const
        {
            unsigned h = 0;
            _fnv_hash_update(h, &record, static_cast<unsigned>(sizeof(record)));
            return static_cast<size_t>(h);
        }
    };
}

#endif
