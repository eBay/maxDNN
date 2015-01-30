/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/

// This is the skeleton of the multiconvolution_64 kernel. It is used to
// generate multiconvolution_64.cubin, then maxas.pl is used to fill in
// the assembly code from multiconvolution_64.sass
#include "maxdnn/ConvolutionIndexesGpu.hpp"
#include "maxdnn/ConvolutionBlockingsGpu.hpp"
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

typedef texture<float4, cudaTextureType1D, cudaReadModeElementType> MyTexture;

MyTexture  texInput(0, cudaFilterModePoint, cudaAddressModeBorder);
MyTexture  texFilters(0, cudaFilterModePoint, cudaAddressModeBorder);

__constant__ maxdnn::ConvolutionIndexesGpu multiconvolution_64_Indexes;
__constant__ maxdnn::ConvolutionBlockingsGpu multiconvolution_64_Blockings;


// Use extern C so C++ doesn't mangle our kernel name
extern "C" __global__ void  multiconvolution_64(float *out,
						int indexOffset,
						int numIndexes,
						int stride,
						int Nb,
						int Nbf,
						int Wstride4,
						int Wo,
						int Ho,
						int No,
						int ldc,
						int padding,
						float alpha,
						int blockingOffset
						, int *diag // diagnostic output
						)
{
  __shared__ float share[2*8*2*64];

  int tid = threadIdx.x;
  // int bx  = blockIdx.x;
  // int by  = blockIdx.y;
  // int blkDimX = blockDim.x;
  // int blkDimY = blockDim.y;
  // int blkDimZ = blockDim.z;
  // int grdDimX = gridDim.x;
  // int grdDimY = gridDim.y;
  // int grdDimZ = gridDim.z;

  int inputIndex = multiconvolution_64_Indexes.imageIndex[tid];

  int i_b = multiconvolution_64_Blockings.i_b[blockingOffset+blockIdx.z];
  int f_b = multiconvolution_64_Blockings.f_b[blockingOffset+blockIdx.z];

  *(float4 *)&share[tid*4] = tex1Dfetch(texInput, i_b*64 + inputIndex);
  *(float4 *)&share[tid*4+256] = tex1Dfetch(texFilters, f_b*64 + tid);
  
  float4 v = tex1Dfetch(texInput, inputIndex+2);

  __syncthreads();

  //    out[tid] = share[tid];

  int im_shuffl_0 = ((tid >> 1) & 7) << 2;
  int flt_shuffl_0 = (((tid & 48) >> 3) | (tid & 1)) << 2;

  out[tid] = share[im_shuffl_0] + v.x;
  out[tid+64] = share[256+flt_shuffl_0] + v.y;
  out[tid+96] = v.z;

  diag[tid] = im_shuffl_0;
}
