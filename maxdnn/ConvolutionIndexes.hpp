/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_ConvolutionIndexes_h
#define maxdnn_ConvolutionIndexes_h

#include "maxdnn/ConvolutionIndexesGpu.hpp"
#include "maxdnn/Exception.hpp"
#include "maxdnn/Shape.hpp"
#include "maxdnn/Hash.hpp"
#include <limits>

namespace maxdnn
{
    class ConvolutionIndexes : public ConvolutionIndexesGpu
    {
        struct Pars
        {
            Pars(const Shape &shape,
                 int Ni,
                 int Sk,
                 int Nld)
                : _shape(shape),
                  _Ni(Ni),
                  _Sk(Sk),
                  _Nld(Nld)
            {
            }

            bool operator==(const Pars &other) const
            {
                return
                    _shape == other._shape &&
                    _Ni == other._Ni &&
                    _Sk == other._Sk &&
                    _Nld == other._Nld;
            }
            
            const Shape _shape;
            const int _Ni;
            const int _Sk;
            const int _Nld;
        };

    public:

        ConvolutionIndexes()
        {
            numArrays = 0;
            memset(arrayOffsets, 0, sizeof(arrayOffsets));
            memset(arraySizes, 0, sizeof(arraySizes));
            for (int index = 0; index < Capacity; ++index) {
                imageIndex[index] = std::numeric_limits<int>::max();
            }
        }

        static int _alignIndex(int index)
        {
            return (index+Alignment-1)/Alignment*Alignment;
        }

        // Get the position of the convolution indexes for the given
        // convolution parameters. Returns true if the indexes were
        // newly added, returns false if the indexes for these
        // parameters were added by a previous call to this method.
        // Not thread safe!
        bool addIndexesForParameters(const Shape &imageShape, int Ni, int Sk, int Nld, int &position)
        {
            Pars pars(imageShape, Ni, Sk, Nld);
            PositionMap::iterator p = _positionMap.find(pars);
            if (p == _positionMap.end()) {
                int pos = _addConvolution(imageShape, Ni, Sk, Nld);
                _positionMap[pars] = pos;
                position = pos;
                return true;
            }
            position = p->second;
            return false;
        }
        
        void clear()
        {
            numArrays = 0;
            _positionMap.clear();
        }
        
    private:

        // Ni: number of input channels
        // Sk: kernel size (Sk X Sk)
        // Nld: number of elements in a single load.
        int _addConvolution(const Shape &imageShape, int Ni, int Sk, int Nld)
        {
            int size = Ni*Sk*Sk;

            MAXDNN_ASSERT(size <= Capacity, BadArgException("layer_size", size, Capacity));

            int index=0;

            if (numArrays > 0) {
                index = _alignIndex(arrayOffsets[numArrays-1] + arraySizes[numArrays-1]);
                if (index + size > Capacity) {
                    // If there is insufficient room in the array, clear it.
                    clear();
                    index = 0;
                }
            }

            // Compute image indices.
            arrayOffsets[numArrays] = index;
            arraySizes[numArrays] = size;
            for (int c=0; c<Ni; ++c) {
                for (int v=0; v<Sk; ++v) {
                    for (int u=0; u<Sk; ++u) {
                        imageIndex[index] = imageShape(c, v, u, 0)/Nld;
                        ++index;
                    }
                }
            }

            return numArrays++;
        }

        typedef std::tr1::unordered_map<Pars, int, Hash<Pars> > PositionMap;
        PositionMap _positionMap;
    };
}

#endif
