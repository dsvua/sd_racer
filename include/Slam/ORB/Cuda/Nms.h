#ifndef JETRACER_CUDA_NMS_KEYPOINTS_H
#define JETRACER_CUDA_NMS_KEYPOINTS_H

#include <vector>
#include <cuda_runtime.h>
#include "../../Defines.h"
#include "../../../Types/Frames.h"

namespace Jetracer
{
    void grid_nms(pRgbdFrame current_frame, TmpData_t &tmp_frame);

} // namespace Jetracer

#endif // JETRACER_CUDA_NMS_KEYPOINTS_H
