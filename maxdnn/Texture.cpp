/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/Texture.hpp"
#include "maxdnn/gpu.h"
#include <string.h>

namespace maxdnn
{
    Texture::Pointer Texture::makeTexture(const float *data, int numElements)
    {
        return Pointer(new Texture(data, numElements));
    }
    
        
    Texture::Pointer Texture::makeTexture4(const float *data, int numElements)
    {
        return Pointer(new Texture(reinterpret_cast<const float4*>(data), numElements/4));
    }
        
    Texture::Pointer Texture::makeTexture(const float4 *data, int numElements)
    {
        return Pointer(new Texture(data, numElements));
    }
    
    Texture::Texture(const float *data, int numElements)
    {
        cudaResourceDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.resType = cudaResourceTypeLinear;
        desc.res.linear.devPtr = const_cast<float *>(data);
        desc.res.linear.desc = cudaCreateChannelDesc<float>();
        desc.res.linear.sizeInBytes = sizeof(float)*numElements;
        
        cudaTextureDesc tdesc;
        memset(&tdesc, 0, sizeof(tdesc));

        CUDA_CHECK(cudaCreateTextureObject(&_texture, &desc, &tdesc, NULL));
    }
        
    Texture::Texture(const float4 *data, int numElements)
    {
        cudaResourceDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.resType = cudaResourceTypeLinear;
        desc.res.linear.devPtr = const_cast<float4 *>(data);
        desc.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        desc.res.linear.sizeInBytes = sizeof(float4)*numElements;
        
        cudaTextureDesc tdesc;
        memset(&tdesc, 0, sizeof(tdesc));
        tdesc.addressMode[0] = cudaAddressModeBorder;
        tdesc.addressMode[1] = cudaAddressModeBorder;
        tdesc.addressMode[2] = cudaAddressModeBorder;

        CUDA_CHECK(cudaCreateTextureObject(&_texture, &desc, &tdesc, NULL));
    }

    Texture::~Texture()
    {
        if (_texture) {
            cudaDestroyTextureObject(_texture);
        }
    }

    Texture::Texture(Texture &)
    {
    }
    

    const Texture& Texture::operator=(const Texture&)
    {
        return *this;
    }
}
