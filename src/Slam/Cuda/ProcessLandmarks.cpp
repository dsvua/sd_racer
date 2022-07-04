// #include "Slam/ORB/OrbDetector.h"
// #include "Slam/ORB/Cuda/Orb.h"
#include "Cuda/CudaCommon.h"
#include "Types/Defines.h"
#include "Types/Frames.h"
#include "Slam/Cuda/ProcessLandmarks.h"

namespace Jetracer
{
    __global__ void copy_visible_landmarks_kenel(PointCoordinates *d_dst_world_coordinates,
                                                 PointCoordinates *d_dst_camera_coordinates,
                                                 float2 *d_dst_camera_image_coordinates,
                                                 uint *d_dst_num_of_measurements,
                                                 uint *d_dst_global_id,
                                                 uint4 *d_dst_descriptor1,
                                                 uint4 *d_dst_descriptor2,

                                                 PointCoordinates *d_src_world_coordinates,
                                                 PointCoordinates *d_src_camera_coordinates,
                                                 float2 *d_src_camera_image_coordinates,
                                                 uint *d_src_num_of_measurements,
                                                 uint *d_src_global_id,
                                                 uint4 *d_src_descriptor1,
                                                 uint4 *d_src_descriptor2,

                                                 CameraMatrix *d_camera_matrix,
                                                 int2 image_resolution,
                                                 int landmarks_num,
                                                 int *d_landmarks_num_visible,
                                                 int maximum_projection_tracking_distance_pixels,
                                                 TransformMatrix3D *d_world_to_camera_guess)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // dereference Eigen matrixes
        CameraMatrix camera_matrix = *d_camera_matrix;
        TransformMatrix3D world_to_camera_guess = *d_world_to_camera_guess;

        if (idx < landmarks_num)
        {
            // Landmark_t landmark = d_landmarks[idx];
            PointCoordinates camera_coordinates_prediction = world_to_camera_guess * d_src_world_coordinates[idx];
            const Vector3 point_in_image_prediction = camera_matrix * camera_coordinates_prediction;
            float2 camera_coordinates;
            camera_coordinates.x = point_in_image_prediction(0, 0);
            camera_coordinates.y = point_in_image_prediction(1, 0);

            if (camera_coordinates.x > 0 - maximum_projection_tracking_distance_pixels &&
                camera_coordinates.x < image_resolution.x + maximum_projection_tracking_distance_pixels &&
                camera_coordinates.y > 0 - maximum_projection_tracking_distance_pixels &&
                camera_coordinates.y < image_resolution.y + maximum_projection_tracking_distance_pixels)
            {
                int visible_idx = atomicAdd(d_landmarks_num_visible, 1);
                d_dst_world_coordinates[visible_idx] = d_src_world_coordinates[idx];
                d_dst_camera_image_coordinates[visible_idx] = camera_coordinates;
                d_dst_num_of_measurements[visible_idx] = d_dst_num_of_measurements[idx];
                d_dst_global_id[visible_idx] = idx;
                d_src_descriptor1[visible_idx] = d_src_descriptor1[idx];
                d_src_descriptor2[visible_idx] = d_src_descriptor2[idx];
            }
        }
    }

    __global__ void match_points_to_landmarks_kernel(float2 *dst_points_pos,
                                                     int *dst_points_num,
                                                     float2 *src_points_pos,
                                                     int *src_points_num,
                                                     int2 *matched_pairs,
                                                     int *matched_pairs_num,
                                                     uint4 dst_descriptors1,
                                                     uint4 dst_descriptors2,
                                                     uint4 src_descriptors1,
                                                     uint4 src_descriptors2,
                                                     float max_distance,
                                                     float max_descriptor_distance)
    {
        float distance_squared = 0;
        float max_distance_squared = max_distance * max_distance;
        float min_distance_squared = max_distance_squared;
        int min_distance_index = -1;
        uint4 dst_descriptor1;
        uint4 dst_descriptor2;
        uint4 src_descriptor1;
        uint4 src_descriptor2;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float2 src_p = src_points_pos[idx];
        if (idx < src_points_num)
        {
            src_descriptor1 = src_descriptors1[idx];
            src_descriptor2 = src_descriptors2[idx];

            for (int dstIdx = 0; dstIdx < dst_points_num)
            {
                float2 dst_p = dst_points_pos[idx];
                dst_descriptor1 = dst_descriptors1[idx];
                dst_descriptor2 = dst_descriptors2[idx];

                float dx = src_p.x - dst_p.x;
                float dy = src_p.y - dst_p.y;
                distance_squared = dx * dx + dy * dy;

                int descriptor_distance = __popc(src_descriptor1.x ^ dst_descriptor1.x) +
                                          __popc(src_descriptor1.y ^ dst_descriptor1.y) +
                                          __popc(src_descriptor1.z ^ dst_descriptor1.z) +
                                          __popc(src_descriptor1.w ^ dst_descriptor1.w);

                int descriptor_distance = descriptor_distance +
                                          __popc(src_descriptor2.x ^ dst_descriptor2.x) +
                                          __popc(src_descriptor2.y ^ dst_descriptor2.y) +
                                          __popc(src_descriptor2.z ^ dst_descriptor2.z) +
                                          __popc(src_descriptor2.w ^ dst_descriptor2.w);

                if (distance_squared < min_distance_squared &&
                    (float)descriptor_distance < max_descriptor_distance)
                {
                    min_distance_index = dstIdx;
                    min_distance_squared = distance_squared;
                }
            }

            if (min_distance_index > -1)
            {
                int matched_idx = atomicAdd(matched_pairs_num, 1);
                matched_pairs[matched_idx] = make_int2(idx, min_distance_index);
            }
        }
    }

    void copy_visible_landmarks(pRgbdFrame current_frame,
                                TmpData_t &tmp_frame,
                                float max_descriptor_distance,
                                float maximum_projection_tracking_distance_pixels)
    {
        if (tmp_frame.worldmap_landmarks.size > 0)
        {
            CameraMatrix *d_camera_matrix;
            checkCudaErrors(cudaMalloc((void **)&d_camera_matrix,
                                       sizeof(CameraMatrix)));
            checkCudaErrors(cudaMemcpy((void *)d_camera_matrix,
                                       (void *)current_frame->camera_matrix.data(),
                                       sizeof(CameraMatrix),
                                       cudaMemcpyHostToDevice));

            TransformMatrix3D *d_world_to_camera_guess;
            checkCudaErrors(cudaMalloc((void **)&d_world_to_camera_guess,
                                       sizeof(TransformMatrix3D)));
            checkCudaErrors(cudaMemcpy((void *)d_world_to_camera_guess,
                                       (void *)current_frame->world_to_camera_guess.data(),
                                       sizeof(TransformMatrix3D),
                                       cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void **)&tmp_frame.d_landmarks_num_visible,
                                       sizeof(TransformMatrix3D)));
            checkCudaErrors(cudaMemcpyAsync((void *)tmp_frame.d_landmarks_num_visible,
                                            (void *)&tmp_frame.visible_landmarks.size,
                                            sizeof(int),
                                            cudaMemcpyHostToDevice,
                                            tmp_frame.stream));

            dim3 threads(CUDA_WARP_SIZE);
            dim3 blocks(calc_block_size(tmp_frame.worldmap_landmarks.size, threads.x));

            copy_visible_landmarks_kenel<<<blocks, threads, 0, tmp_frame.stream>>>(tmp_frame.visible_landmarks.d_world_coordinates,
                                                                                   tmp_frame.visible_landmarks.d_camera_coordinates,
                                                                                   tmp_frame.visible_landmarks.d_camera_image_coordinates,
                                                                                   tmp_frame.visible_landmarks.d_num_of_measurements,
                                                                                   tmp_frame.visible_landmarks.d_global_id,
                                                                                   tmp_frame.visible_landmarks.d_descriptor1,
                                                                                   tmp_frame.visible_landmarks.d_descriptor2,

                                                                                   tmp_frame.worldmap_landmarks.d_world_coordinates,
                                                                                   tmp_frame.worldmap_landmarks.d_camera_coordinates,
                                                                                   tmp_frame.worldmap_landmarks.d_camera_image_coordinates,
                                                                                   tmp_frame.worldmap_landmarks.d_num_of_measurements,
                                                                                   tmp_frame.worldmap_landmarks.d_global_id,
                                                                                   tmp_frame.worldmap_landmarks.d_descriptor1,
                                                                                   tmp_frame.worldmap_landmarks.d_descriptor2,

                                                                                   d_camera_matrix,
                                                                                   current_frame->rgb_image_resolution,
                                                                                   tmp_frame.worldmap_landmarks.size,
                                                                                   tmp_frame.d_landmarks_num_visible,
                                                                                   maximum_projection_tracking_distance_pixels,
                                                                                   d_world_to_camera_guess);

            // TODO: check if I need it actually
            checkCudaErrors(cudaMemcpyAsync((void *)&tmp_frame.visible_landmarks.size,
                                            (void *)tmp_frame.d_landmarks_num_visible,
                                            sizeof(int),
                                            cudaMemcpyDeviceToHost,
                                            tmp_frame.stream));

            // preparing for matching landpoints
            checkCudaErrors(cudaMalloc((void **)&current_frame->d_num_of_matched_landmarks,
                                       sizeof(int2)));
            checkCudaErrors(cudaMemcpyAsync((void *)current_frame->d_num_of_matched_landmarks,
                                            (void *)&current_frame->h_num_of_matched_landmarks,
                                            sizeof(int2),
                                            cudaMemcpyHostToDevice,
                                            tmp_frame.stream));

            // matching current frame points to landpoints
            blocks(calc_block_size(current_frame->d_keypoints_num, threads.x));
            match_points_to_landmarks_kernel<<<blocks, threads, 0, tmp_frame.stream>>>(tmp_frame.visible_landmarks.d_camera_image_coordinates,
                                                                                       tmp_frame.d_landmarks_num_visible,
                                                                                       current_frame->d_keypoints_pos,
                                                                                       current_frame->d_keypoints_num,
                                                                                       current_frame->d_matched_landmarks,
                                                                                       current_frame->d_num_of_matched_landmarks,
                                                                                       tmp_frame.visible_landmarks.d_descriptors1,
                                                                                       tmp_frame.visible_landmarks.d_descriptors2,
                                                                                       current_frame->d_descriptors1,
                                                                                       current_frame->d_descriptors2,
                                                                                       maximum_projection_tracking_distance_pixels,
                                                                                       max_descriptor_distance);
        }
    }
} // namespace Jetracer
