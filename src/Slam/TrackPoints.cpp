#include "Slam/SlamLoop.h"
#include "Cuda/CudaCommon.h"

namespace Jetracer
{
    void SlamLoop::trackPoints(pRgbdFrame current_frame)
    {
        if (tmp_frame.previous_frame)
        {
            // track points between frames
        }

        // robot_to_world = current_frame->robot_to_world;
    }

} // namespace Jetracer
