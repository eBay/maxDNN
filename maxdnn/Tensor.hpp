/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#ifndef maxdnn_Tensor_h
#define maxdnn_Tensor_h

#include "maxdnn/Shape.hpp"
#include "maxdnn/BlockData.hpp"
#include "maxdnn/CppData.hpp"
#include "maxdnn/Exception.hpp"
#include "maxdnn/TensorRef.hpp"
#include <tr1/memory>
#include <iostream>

namespace maxdnn
{
    typedef float Float;

    template<typename Scalar>
    class Tensor
    {
        void _create()
        {
            size_t size = _shape.getSize<Scalar>();
            if (!_data) {
                _data = CppData::make(size);
            } else if (_data->getSize() < size) {
                _data = _data->clone(size);
            }
            _T = static_cast<Scalar *>(_data->getData());
        }
        
    public:

        typedef std::tr1::shared_ptr<Tensor<Scalar> > Pointer;

        typedef TensorRef<Scalar> Ref;

        Tensor()
            : _shape(0, 0, 0, 0),
              _T(0)
        {
        }
        
        Tensor(const Shape &shape, Data::Pointer data=Data::Pointer())
            : _shape(shape),
              _data(data)
        {
            _create();
        }

        Tensor(int K, int L, int M, int N, Data::Pointer data=Data::Pointer())
            : _shape(K, L, M, N),
              _data(data)
        {
            _create();
        }

        Tensor(Tensor<Scalar> &other, Data::Pointer data=Data::Pointer())
            : _shape(other._shape),
              _data(data)
        {
            _create();
            copy(other);
        }
        
        Tensor(BlockData::Pointer blockData)
            : _shape(blockData->getShape()),
              _data(blockData)
        {
            _create();
        }

        ~Tensor()
        {
        }
        
        void create(const Shape &shape, Data::Pointer data=Data::Pointer())
        {
            _shape = shape;
            _data = data;
            _create();
        }

        void create(int K, int L, int M, int N, Data::Pointer data=Data::Pointer())
        {
            create(Shape(K, L, M, N), data);
        }
        
        void create(BlockData::Pointer data)
        {
            _shape = data->getShape();
            _data = data;
            _create();
        }

        TensorRef<Scalar> getRef() const
        {
            return TensorRef<Scalar>(_shape, const_cast<Scalar *>(_T));
        }
    
        const Shape &getShape() const { return _shape; } 

        size_t getSize() const 
        {
            return _shape.getSize<Scalar>();
        }
        
        size_t getNumElements() const
        {
            return _shape.getNumElements();
        }

        TensorRef<Scalar> window(int g=0, int h=0, int i=0, int j=0,
                                 int K=0, int L=0, int M=0, int N=0)
        {
            return getRef().window(g, h, i, j, K, L, M, N);
        }
        
        // Pointer window(int g=0, int h=0, int i=0, int j=0,
        //                int K=0, int L=0, int M=0, int N=0)
        // {
        //     Tensor<Float> *W = new Tensor(_shape, _data);
        //     W->_T = &((*this)(g, h, i, j));
        //     W->_shape.K = K;
        //     W->_shape.L = L;
        //     W->_shape.M = M;
        //     W->_shape.N = N;
        //     return Tensor<Float>::Pointer(W);
        // }
            
        const Scalar &operator()(int g=0, int h=0, int i=0, int j=0) const
        {
            return _T[_shape(g, h, i, j)];
        }
        
        Scalar &operator()(int g=0, int h=0, int i=0, int j=0) 
        {
            return _T[_shape(g, h, i, j)];
        }

        const Scalar &operator[](int i) const 
        {
            return _T[i];
        }
        
        Scalar &operator[](int i)
        {
            return _T[i];
        }
        
        Data::Pointer getData() { return _data; }

        const Data::Pointer &getData() const { return _data; }

        bool operator==(const Tensor &other) const
        {
            if (_shape==other._shape) {
                return (memcmp(_T, other._T, _shape.getSize<Scalar>()) == 0);
            } else if (_shape.sameDimensionsAs(other._shape)) {
                for (int k=0; k<_shape.K; ++k) {
                    for (int l=0; l<_shape.L; ++l) {
                        for (int m=0; m<_shape.M; ++m) {
                            if (memcmp(&(*this)(k,l,m,0), &other(k,l,m,0), _shape.N) != 0) {
                                return false;
                            }
                        }
                    }
                }
                return true;
            } else {
                return false;
            }
        }

        void copy(const Tensor &src)
        {
            MAXDNN_ASSERT_SIZE_MATCH(src.getShape(), _shape, "Tensor");
            _data->copy(*src.getData());
        }

        void fillWithZeros()
        {
            memset(_T, 0, _shape.getSize<Scalar>());
        }
        
    private:

        Shape _shape;
        Data::Pointer _data;
        Scalar *_T;
    };

    template<typename Scalar>
    inline
    std::ostream &operator<<(std::ostream &s, const Tensor<Scalar> &T)
    {
        const Shape &shape = T.getShape();
        const int K = shape.K;
        const int L = shape.L;
        const int M = shape.M;
        const int N = shape.N;
        
        s << "[\n";
        for(int k=0; k<K; ++k) {
            s << "    [\n";
            for (int l=0; l<L; ++l) {
                s << "        [\n";
                for (int m=0; m<M; ++m) {
                    s << "            [";
                    for (int n=0; n<N; ++n) {
                        s << T(k,l,m,n);
                        if (n<N-1) {
                            s << " ";
                        }
                    }
                    s << "]\n";
                }
                s << "        ]\n";
            }
            s << "    ]\n";
        }
        s << "]\n";
        return s;
    }
}

#endif
