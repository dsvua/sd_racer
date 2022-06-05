#ifndef JETRACER_CUDA_ORB_DETECTOR_H
#define JETRACER_CUDA_ORB_DETECTOR_H

#include <cuda_runtime.h>

#include "Cuda/Orb.h"
#include "Cuda/Fast.h"

namespace Jetracer
{
    void detectOrbs(pRgbdFrame current_frame, TmpData_t &tmp_frame);

} // namespace Jetracer

#endif // JETRACER_CUDA_ORB_DETECTOR_H
