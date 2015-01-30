/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_ConvolutionBlockingsGpu_h
#define maxdnn_ConvolutionBlockingsGpu_h

namespace maxdnn
{
    struct ConvolutionBlockingsGpu
    {
        enum
        {
            Alignment = 16,
            Capacity = 64,
            MaxArrays = 32
        };
        
        int numArrays;
        int arrayOffsets[MaxArrays];
        int arraySizes[MaxArrays];
            
        // z-block to image block mapping.
        int i_b[Capacity];

        // z-block to filter block mapping.
        int f_b[Capacity];
    };
}

#endif


