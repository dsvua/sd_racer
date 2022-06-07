#include "Slam/ORB/OrbDetector.h"
#include "Slam/ORB/Cuda/Orb.h"
#include "Cuda/CudaCommon.h"
#include "Types/Defines.h"

namespace Jetracer
{
    void detectOrbs(pRgbdFrame current_frame, TmpData_t &tmp_frame, float min_score)
    {
        if (!tmp_frame.d_corner_lut)
        {
            checkCudaErrors(cudaMalloc((void **)&tmp_frame.d_corner_lut, 64 * 1024));
            fast_gpu_calculate_lut(tmp_frame.d_corner_lut, FAST_MIN_ARC_LENGTH, tmp_frame.stream);
        }

        if (!tmp_frame.d_keypoint_response)
        {
            tmp_frame.max_keypoints_num = calc_block_size(current_frame->rgb_image_resolution.x, CELL_SIZE_WIDTH) * calc_block_size(current_frame->rgb_image_resolution.y, CELL_SIZE_HEIGHT);
            checkCudaErrors(cudaMalloc((void **)&tmp_frame.d_keypoints_angle,
                                       tmp_frame.max_keypoints_num * sizeof(float)));

            checkCudaErrors(cudaMallocPitch((void **)&tmp_frame.d_keypoint_response,
                                            &tmp_frame.d_keypoint_response_pitch,
                                            current_frame->rgb_image_resolution.x * sizeof(float),
                                            current_frame->rgb_image_resolution.y));
            checkCudaErrors(cudaMalloc((void **)&tmp_frame.d_descriptors,
                                       tmp_frame.max_keypoints_num * 32 * sizeof(unsigned char)));
            checkCudaErrors(cudaMalloc((void **)&tmp_frame.d_keypoints_pos,
                                       tmp_frame.max_keypoints_num * sizeof(float2)));
            checkCudaErrors(cudaMalloc((void **)&tmp_frame.d_keypoints_score,
                                       tmp_frame.max_keypoints_num * sizeof(float)));
        }

        checkCudaErrors(cudaMalloc((void **)&current_frame->d_descriptors,
                                   tmp_frame.max_keypoints_num * 32 * sizeof(unsigned char)));
        checkCudaErrors(cudaMalloc((void **)&current_frame->d_keypoints_pos,
                                   tmp_frame.max_keypoints_num * sizeof(float2)));
        checkCudaErrors(cudaMalloc((void **)&current_frame->d_keypoints_score,
                                   tmp_frame.max_keypoints_num * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&current_frame->d_keypoints_num, sizeof(int)));

        fast_detect(current_frame, tmp_frame);
        compute_fast_angle(current_frame, tmp_frame);
        calc_orb(current_frame, tmp_frame);
        filter_keypoints(current_frame, tmp_frame, min_score);
    }

} // namespace Jetracer
