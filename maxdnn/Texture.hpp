/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Texture_h
#define maxdnn_Texture_h

#include <cuda_runtime.h>
#include <tr1/memory>

namespace maxdnn
{
    class Texture
    {
    public:

        typedef std::tr1::shared_ptr<Texture> Pointer;
        
        static Pointer makeTexture(const float *data, int numElements);
        
        static Pointer makeTexture4(const float *data, int numElements);
        
        static Pointer makeTexture(const float4 *data, int numElements);

        cudaTextureObject_t getTexture() 
        {
            return _texture;
        }
        
        ~Texture();

    private:

        Texture(const float *data, int numElements);
        
        Texture(const float4 *data, int numElements);

        Texture(Texture &);

        const Texture& operator=(const Texture&);

        cudaTextureObject_t _texture;
    };
}


#endif
