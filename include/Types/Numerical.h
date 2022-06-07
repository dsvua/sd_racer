#ifndef JETRACER_CUDA_NUMERICAL_TYPES_H
#define JETRACER_CUDA_NUMERICAL_TYPES_H

#include <cuda_runtime.h>

namespace Jetracer

{
    struct __builtin_align__(32) ulong8
    {
        unsigned long int x, y, z, w, x1, y2, z2, w2;
    };
}

#endif // JETRACER_CUDA_NUMERICAL_TYPES_H