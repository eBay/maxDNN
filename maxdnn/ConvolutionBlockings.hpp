/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_ConvolutionBlockings_h
#define maxdnn_ConvolutionBlockings_h

#include "maxdnn/ConvolutionBlockingsGpu.hpp"
#include "maxdnn/Exception.hpp"
#include "maxdnn/Shape.hpp"
#include "maxdnn/Hash.hpp"
#include <stdint.h>
#include <tr1/unordered_map>
#include <limits>

namespace maxdnn
{
    class ConvolutionBlockings : public ConvolutionBlockingsGpu
    {
        struct Pars
        {
            Pars(int No,
                 int Nb)
                : _No(No),
                  _Nb(Nb)
            {
            }

            bool operator==(const Pars &other) const
            {
                return
                    _No == other._No &&
                    _Nb == other._Nb;
            }
            
            const int _No;
            const int _Nb;
        };
        
    public:

        ConvolutionBlockings()
        {
            numArrays = 0;
            memset(arrayOffsets, 0, sizeof(arrayOffsets));
            memset(arraySizes, 0, sizeof(arraySizes));
            for (int index = 0; index < Capacity; ++index) {
                i_b[index] = std::numeric_limits<int>::max();
                f_b[index] = std::numeric_limits<int>::max();
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
        bool addBlockingsForParameters(int No, int Nb, int &position)
        {
            Pars pars(No, Nb);
            PositionMap::iterator p = _positionMap.find(pars);
            if (p == _positionMap.end()) {
                int pos = _addBlocking(No, Nb);
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

        int _addBlocking(int No, int Nb)
        {
            // Compute image and filter block mappings.
            const int Nbf = (No+63)/64;
            const int size = ((No+63)/64)*((Nb+63)/64);
            
            MAXDNN_ASSERT(size <= Capacity, BadArgException("size", size, Capacity));

            int index=0;

            if (numArrays > 0) {
                // TODO: Why is the caching broken?
//                index = _alignIndex(arrayOffsets[numArrays-1] + arraySizes[numArrays-1]);
//                if (index + size > Capacity) {
//                    // If there is insufficient room in the array, clear it.
                    clear();
                    index = 0;
//                }
            }

            arrayOffsets[numArrays] = index;
            arraySizes[numArrays] = size;
            for (int bz=0; bz<size; ++bz, ++index) {
                i_b[index] = bz/Nbf;
                f_b[index] = bz%Nbf;
            }

            return numArrays++;
        }

        typedef std::tr1::unordered_map<Pars, int, Hash<Pars> > PositionMap;
        PositionMap _positionMap;
    };
}

#endif
