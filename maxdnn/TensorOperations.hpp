/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_TensorOperations_h
#define maxdnn_TensorOperations_h

#include "maxdnn/Tensor.hpp"
#include "maxdnn/Random.hpp"

namespace maxdnn
{
    template<typename Scalar>
    inline
    void setUniformRandom(Tensor<Scalar> &T, Random& ran, Scalar min, Scalar max)
    {
        for (int k=0; k<T.getShape().K; ++k) {
            for (int l=0; l<T.getShape().L; ++l) {
                for (int m=0; m<T.getShape().M; ++m) {
                    for (int n=0; n<T.getShape().N; ++n) {
                        T(k,l,m,n) = ran.uniform(min, max);
                    }
                }
            }
        }
    }

    template<typename Scalar>
    inline
    void set(Tensor<Scalar> &T, Scalar val)
    {
        for (int k=0; k<T.getShape().K; ++k) {
            for (int l=0; l<T.getShape().L; ++l) {
                for (int m=0; m<T.getShape().M; ++m) {
                    for (int n=0; n<T.getShape().N; ++n) {
                        T(k,l,m,n) = val;
                    }
                }
            }
        }
    }
}

#endif
