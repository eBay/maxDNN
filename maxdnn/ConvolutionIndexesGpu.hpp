/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_ConvolutionIndexesGpu_h
#define maxdnn_ConvolutionIndexesGpu_h

namespace maxdnn
{
    struct ConvolutionIndexesGpu
    {
        enum
        {
            Alignment = 16,
            Capacity = 16000, // ints
            MaxArrays = 32            
        };

        int numArrays;
        int arrayOffsets[MaxArrays];
        int arraySizes[MaxArrays];
        int imageIndex[Capacity];
    };
}

#endif
