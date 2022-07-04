#ifndef JETRACER_PROCESS_LANDMARKS_H
#define JETRACER_PROCESS_LANDMARKS_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "../../Types/Frames.h"
#include "../../Types/Landmarks.h"

namespace Jetracer
{
    void copy_visible_landmarks(pRgbdFrame current_frame,
                                TmpData_t &tmp_frame,
                                float max_descriptor_distance,
                                float maximum_projection_tracking_distance_pixels);
}

#endif // JETRACER_PROCESS_LANDMARKS_H
