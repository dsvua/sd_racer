#include "Slam/SlamLoop.h"
#include "Cuda/CudaCommon.h"
#include "Slam/Cuda/ProcessLandmarks.h"

namespace Jetracer
{
    void SlamLoop::trackPoints(pRgbdFrame current_frame)
    {
        if (tmp_frame.previous_frame)
        {
            // track points between frames
            switch (tmp_frame.current_robot_status)
            {
            case RobotStatus::Localizing:
            {
                break;
            }
            default:
                break;
            }
        }

        // find all landmarks that are visible in current frame
        if (tmp_frame.visible_landmarks.allocated_size < tmp_frame.worldmap_landmarks.size)
            tmp_frame.visible_landmarks.resize(tmp_frame.worldmap_landmarks.size, tmp_frame.stream);

        tmp_frame.visible_landmarks.size = 0;

        if (tmp_frame.worldmap_landmarks.allocated_size - tmp_frame.worldmap_landmarks.size < tmp_frame.max_keypoints_num)
            tmp_frame.worldmap_landmarks.expand(tmp_frame.stream);

        copy_visible_landmarks(current_frame,
                               tmp_frame,
                               _ctx->maximum_descriptor_distance_tracking,
                               _ctx->maximum_projection_tracking_distance_pixels);

        // robot_to_world = current_frame->robot_to_world;
    }

} // namespace Jetracer
