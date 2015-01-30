/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_IndexesInstance_h
#define maxdnn_IndexesInstance_h

#include "maxdnn/ConvolutionIndexes.hpp"
#include "maxdnn/Shape.hpp"

namespace maxdnn
{
  class IndexesInstance
  {
  public:

    static IndexesInstance &getInstance()
    {
      // TODO: Not thread safe.
      static IndexesInstance indexesInstance;
      return indexesInstance;
    }

    void getOffsetAndSize(const Shape &imagesShape, int Ni, int Sk, int Nld, int &offset, int &size)
    {
      int indexesPosition = getConvolutionIndexesPosition(imagesShape, Ni, Sk, Nld);
      offset = _indexes.arrayOffsets[indexesPosition];
      size = _indexes.arraySizes[indexesPosition];
    }

    int getConvolutionIndexesPosition(const Shape &imageShape, int Ni, int Sk, int Nld)
    {
      int position;
      if (_indexes.addIndexesForParameters(imageShape, Ni, Sk, Nld, position)) {
        // First time we encountered the parameters. Must update the "constant" gpu data.
        CUDA_CHECK(cudaMemcpyToSymbol(c_Indexes, 
                                      static_cast<ConvolutionIndexesGpu*>(&_indexes),
                                      sizeof(ConvolutionIndexesGpu)));
      }
      return position;
    }

  private:
    ConvolutionIndexes _indexes;
  };
}

#endif
