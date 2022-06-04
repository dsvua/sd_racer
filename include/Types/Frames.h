#ifndef JETRACER_FRAME_TYPES_THREAD_H
#define JETRACER_FRAME_TYPES_THREAD_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <eigen3/Eigen/Eigen>
#include "../RealSense/RealSenseD400.h"

namespace Jetracer
{

#pragma once

    typedef struct slam_frame_callback
    {
        std::shared_ptr<rgbd_frame_t> rgbd_frame;
        bool image_ready_for_process;
        bool exit_gpu_pipeline;
        std::thread gpu_thread;
        std::mutex thread_mutex;
        std::condition_variable thread_cv;

    } slam_frame_callback_t;

    enum class frame_type
    {
        RGBD,
        STEREO,
        GYRO,
        ACCEL,
    };

    struct base_frame
    {
        frame_type frame_type;
        pEvent event;

    }

    typedef struct slam_frame : base_frame
    {
        unsigned char *image;
        size_t image_length;
        std::shared_ptr<double[]> h_points;
        std::shared_ptr<uint32_t[]> h_descriptors;

        uint32_t *d_descriptors_left;
        uint32_t *d_descriptors_right;
        float2 *d_pos_left;
        float2 *d_pos_right;
        int2 *d_matching_pairs;

        int keypoints_count;
        int h_valid_keypoints_num;
        int h_matched_keypoints_num;
        float3 theta;
        std::shared_ptr<rgbd_frame_t> rgbd_frame;

        Eigen::Matrix4d T_c2w;
        Eigen::Matrix4d T_w2c;

        ~slam_frame()
        {
            if (image)
                free(image);
            if (d_descriptors_left)
                checkCudaErrors(cudaFree(d_descriptors_left));
            if (d_descriptors_right)
                checkCudaErrors(cudaFree(d_descriptors_right));
            if (d_pos_left)
                checkCudaErrors(cudaFree(d_pos_left));
            if (d_pos_right)
                checkCudaErrors(cudaFree(d_pos_right));
            if (d_matching_pairs)
                checkCudaErrors(cudaFree(d_matching_pairs));
        }

    } slam_frame_t;

    typedef struct rgbd_frame
    {
        // rs2::depth_frame depth_frame = rs2::frame{};
        // rs2::video_frame rgb_frame = rs2::frame{};
        // uint16_t *depth_image;
        unsigned char *rgb_image;
        unsigned char *ir_image_left;
        unsigned char *ir_image_right;

        float3 theta; // gyro and accel computed angles for this frame

        double timestamp;
        // unsigned long long depth_frame_id;
        unsigned long long rgb_frame_id;
        unsigned long long ir_frame_id;

        // int depth_image_size;
        int rgb_image_size;
        int ir_image_size;

        // rs2_intrinsics depth_intristics;
        // rs2_intrinsics rgb_intristics;
        rs2_intrinsics ir_intristics_left;
        rs2_intrinsics ir_intristics_right;
        // rs2_extrinsics extrinsics;
        // float depth_scale;

        // float depth_scale;

        std::chrono::_V2::system_clock::time_point RS400_callback;
        std::chrono::_V2::system_clock::time_point GPU_scheduled;
        std::chrono::_V2::system_clock::time_point GPU_callback;
        std::chrono::_V2::system_clock::time_point GPU_EventSent;

        ~rgbd_frame()
        {
            // if (depth_image)
            //     free(depth_image);
            if (rgb_image)
                free(rgb_image);
            if (ir_image_left)
                free(ir_image_left);
            if (ir_image_right)
                free(ir_image_right);
        }

    } rgbd_frame_t;

    typedef struct imu_frame
    {
        rs2_vector motion_data;
        double timestamp;
        rs2_stream frame_type;
    } imu_frame_t;

}

#endif // JETRACER_FRAME_TYPES_THREAD_H
