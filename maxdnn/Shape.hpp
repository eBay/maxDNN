/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Shape_h
#define maxdnn_Shape_h

#include "maxdnn/gpu.h"
#include <memory.h>
#include <iosfwd>


namespace maxdnn
{
    struct Shape
    {
    public:

        MAXDNN_METHOD Shape()
        {
        }

        MAXDNN_METHOD Shape(int K, int L, int M, int N,
                         int strideM=0,
                         int strideL=0,
                         int strideK=0)
        {
            this->K = K;
            this->L = L;
            this->M = M;
            this->N = N;
            this->strideM = (strideM==0) ? N : strideM;
            this->strideL = (strideL==0) ? M*this->strideM : strideL;
            this->strideK = (strideK==0) ? L*this->strideL : strideK;
        }

        MAXDNN_METHOD ~Shape()
        {
        }

        MAXDNN_METHOD const Shape &operator=(const Shape &other)
        {
            if (this != &other) {
                K = other.K;
                L = other.L;
                M = other.M;
                N = other.N;
                strideK = other.strideK;
                strideL = other.strideL;
                strideM = other.strideM;
            }
            return *this;
        }

        MAXDNN_METHOD Shape(const Shape &other)
        {
            *this = other;
        }
        
        MAXDNN_METHOD int operator()(int g=0, int h=0, int i=0, int j=0) const
        {
            return g*strideK + h*strideL + i*strideM + j;
        }

        template<class Scalar>
        MAXDNN_METHOD size_t getSize() const { return sizeof(Scalar)*K*strideK; }

        MAXDNN_METHOD size_t getNumElements() const { return K*L*M*N; }

        MAXDNN_METHOD int getNumImages() const { return K; } 
        MAXDNN_METHOD int getNumChannels() const { return L; } 
        MAXDNN_METHOD int getNumRows() const { return M; } 
        MAXDNN_METHOD int getNumColumns() const { return N; } 
        
        MAXDNN_METHOD void setDefaultStrides()
        {
            this->strideM = N;
            this->strideL = M*this->strideM;
            this->strideK = L*this->strideL;
        }

        MAXDNN_METHOD bool operator==(const Shape &other) const
        {
            return
                K==other.K &&
                L==other.L &&
                M==other.M &&
                N==other.N &&
                strideK==other.strideK &&
                strideL==other.strideL &&
                strideM==other.strideM;
        }

        MAXDNN_METHOD bool sameDimensionsAs(const Shape &other) const
        {
            return
                K==other.K &&
                L==other.L &&
                M==other.M &&
                N==other.N;
        }
        
        MAXDNN_METHOD bool operator!=(const Shape &shape) const
        {
            return !(*this == shape);
        }

        MAXDNN_METHOD std::string getDescription() const;
        
        MAXDNN_METHOD static Shape Zero() 
        {
            return Shape(0, 0, 0, 0);
        }
        
        // images
        int K;

        // channels
        int L;

        // rows
        int M;
        
        // columns
        int N;

        int strideK;
        int strideL;
        int strideM;
    };

    std::ostream & operator<<(std::ostream &s, const Shape &shape);    
}


#endif
