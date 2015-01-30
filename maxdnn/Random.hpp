/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Random_h
#define maxdnn_Random_h

#include <cstdlib>

namespace maxdnn
{
    class Random
    {
    public:

        Random()
        {
        }

        Random(unsigned seed)
        {
            _seed = seed;
        }

        void setSeed(unsigned seed)
        {
            _seed = seed;
        }
        
        double uniform()
        {
            return rand_r(&_seed) / (RAND_MAX+1.);
        }

        double uniform(double min, double max)
        {
            return min + uniform()*(max - min);
        }

    private:

        unsigned _seed;
    };
}

#endif
