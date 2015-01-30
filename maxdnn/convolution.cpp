/*
Copyright (c) 2014 eBay Software Foundation
Licensed under the MIT License
*/
#include "maxdnn/convolution.hpp"
#include "maxdnn/Exception.hpp"
#include <algorithm>
using namespace std;

namespace maxdnn
{
    // Slow but accurate convolution. Used to generate reference output.
    void convolve_cpu(const Tensor<Float> &images,
                      const Tensor<Float> &filters,
                      Tensor<Float> &maps,
                      unsigned stride,
                      unsigned padding,
                      int maxPixels)
    {
        const int No = maps.getShape().K;
        const int Ho = maps.getShape().L;
        const int Wo = maps.getShape().M;
        const int Nb = maps.getShape().N;

        const int Ni = filters.getShape().K;
        const int Hk = filters.getShape().L;
        const int Wk = filters.getShape().M;
        MAXDNN_ASSERT_SIZE_MATCH(No, (int)filters.getShape().N, "No");
    
        MAXDNN_ASSERT_SIZE_MATCH(Ni, (int)images.getShape().K, "Ni");
        const int Hi = images.getShape().L;
        const int Wi = images.getShape().M;
        MAXDNN_ASSERT_SIZE_MATCH(Nb, (int)images.getShape().N, "Nb");

        memset(maps.getData()->getData(), 0, maps.getData()->getSize());

        for (int y=0; y<Ho; ++y) {
            for (int x=0; x< Wo; ++x) {
                int pixels = 0;
                int v_min = y*stride-padding;
                int u_min = x*stride-padding;
                int v_max = min(Hi, v_min+Hk);
                int u_max = min(Wi, u_min+Wk);
                for (int c=0; c<Ni; ++c) {
                    for (int v=max(0, v_min); v<v_max && (maxPixels == 0 || pixels < maxPixels); ++v) {
                        for (int u=max(0, u_min); u<u_max; ++u) {
                            for (int f=0; f<No; ++f) {
                                for (int i=0; i<Nb; ++i) {
                                    maps(f, y, x, i) += filters(c, v-v_min, u-u_min, f)*images(c, v, u, i);
                                }
                            }
                            if (maxPixels > 0 && ++pixels >= maxPixels) {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}
