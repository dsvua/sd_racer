#ifndef JETRACER_RGB_TO_GRAYSCALE_H
#define JETRACER_RGB_TO_GRAYSCALE_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../../../Types/Frames.h"

namespace Jetracer
{
    void rgb_to_grayscale(pRgbdFrame current_frame, TmpData_t &tmp_frame);
}

#endif // JETRACER_RGB_TO_GRAYSCALE_H
