/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_TensorRef_h
#define maxdnn_TensorRef_h

#include "maxdnn/Shape.hpp"

namespace maxdnn
{
    typedef float Float;

    template<typename Scalar>
    class TensorRef
    {
    public:

        MAXDNN_METHOD TensorRef(const Shape &shape, Scalar *data)
            : _shape(shape),
              _T(data)
        {
        }

        MAXDNN_METHOD TensorRef(int K, int L, int M, int N, Scalar *data)
            : _shape(K, L, M, N),
              _T(data)
        {
        }

        MAXDNN_METHOD const TensorRef &operator=(const TensorRef &other)
        {
            if (this != &other) {
                _shape = other._shape;
                _T = other._T;
            }
            return *this;
        }
        
        MAXDNN_METHOD TensorRef(const TensorRef &other)
        {
            *this = other;
        }
        
        MAXDNN_METHOD ~TensorRef()
        {
        }
        
        MAXDNN_METHOD const Shape &getShape() const { return _shape; } 
        
        MAXDNN_METHOD const __restrict Scalar &operator()(int g=0, int h=0, int i=0, int j=0) const
        {
            return _T[_shape(g, h, i, j)];
        }
        
        MAXDNN_METHOD Scalar &operator()(int g=0, int h=0, int i=0, int j=0) 
        {
            return _T[_shape(g, h, i, j)];
        }

        MAXDNN_METHOD const __restrict Scalar &operator[](int i) const 
        {
            return _T[i];
        }
        
        MAXDNN_METHOD Scalar &operator[](int i)
        {
            return _T[i];
        }
        
        MAXDNN_METHOD TensorRef<Float> window(int g=0, int h=0, int i=0, int j=0,
                                           int K=0, int L=0, int M=0, int N=0) const
        {
            TensorRef<Float> W(_shape, _T);
            W._T = const_cast<Float *>(&((*this)(g, h, i, j)));
            W._shape.K = K;
            W._shape.L = L;
            W._shape.M = M;
            W._shape.N = N;
            return W;
        }

#ifdef __CUDACC__
        bool operator==(const TensorRef &other) const
        {
            return
                (_shape==other._shape) &&
                (memcmp(_T, other._T, _shape.getSize<Scalar>()) == 0);
        }
#endif

    private:

        Shape _shape;
        Scalar *_T;
    };
}

#endif
