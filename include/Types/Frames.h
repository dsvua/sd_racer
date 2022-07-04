#ifndef JETRACER_FRAME_TYPES_H
#define JETRACER_FRAME_TYPES_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <eigen3/Eigen/Eigen>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "Defines.h"
#include "Landmarks.h"

namespace Jetracer
{

#pragma once

    enum class RobotStatus
    {
        Localizing,
        Tracking
    };

    enum class FrameType
    {
        RGBD,
        STEREO,
        GYRO,
        ACCEL,
    };

    typedef struct BaseFrame
    {
        FrameType frame_type;
        double timestamp;

    } BaseFrame_t;

    typedef std::shared_ptr<Jetracer::BaseFrame> pBaseFrame;

    typedef struct TmpData
    {
        cudaStream_t stream;

        // For RealSense camera to align depth
        std::shared_ptr<rs2_intrinsics> d_rgb_intrinsics;
        std::shared_ptr<rs2_intrinsics> d_depth_intrinsics;
        std::shared_ptr<rs2_extrinsics> d_depth_other_extrinsics;
        std::shared_ptr<int2> d_pixel_map;

        // For ORB
        unsigned char *d_corner_lut;
        std::size_t d_keypoint_response_pitch;
        float *d_keypoint_response;
        float *d_keypoints_angle;
        const float threshold = 200;

        float2 *d_keypoints_pos;
        float *d_keypoints_score;
        unsigned char *d_descriptors;

        int max_keypoints_num = 0;

        // Landmarks frame_points; // they are not landmarks yet
        Landmarks worldmap_landmarks;
        Landmarks visible_landmarks;
        int *d_landmarks_num_visible;

        // tracking
        TransformMatrix3D world_to_camera_guess = TransformMatrix3D::Identity();
        TransformMatrix3D robot_to_world = TransformMatrix3D::Identity();

        RobotStatus current_robot_status = RobotStatus::Localizing;
        pBaseFrame previous_frame;

        ~TmpData()
        {
            if (d_corner_lut)
                checkCudaErrors(cudaFree(d_corner_lut));
            if (d_keypoint_response)
                checkCudaErrors(cudaFree(d_keypoint_response));
            if (d_keypoints_angle)
                checkCudaErrors(cudaFree(d_keypoints_angle));
            if (d_keypoints_pos)
                checkCudaErrors(cudaFree(d_keypoints_pos));
            if (d_keypoints_score)
                checkCudaErrors(cudaFree(d_keypoints_score));
            if (d_descriptors)
                checkCudaErrors(cudaFree(d_descriptors));
        }

    } TmpData_t;

    typedef struct ImageFrame : BaseFrame
    {
        FrameType frame_type;
        double timestamp;
        Landmarks frame_points;

        CameraMatrix camera_matrix;

        bool keypoints_detected = false;
        unsigned int keypoints_num = 0;
        unsigned int *d_keypoints_num;

        // For ORB
        float2 *d_keypoints_pos;
        float *d_keypoints_score;

        // split descriptor in half, since I can read only
        // 16 bytes in one read and I want coalsced read of descriptors
        unsigned char *d_descriptors1;
        unsigned char *d_descriptors2;

        int2 *d_matched_landmarks;
        int2 *d_matched_points;
        int *d_num_of_matched_landmarks;
        int *d_num_of_matched_points;
        int h_num_of_matched_landmarks = 0;
        int h_num_of_matched_points = 0;

        PointCoordinates *d_points;

        TransformMatrix3D robot_to_world = TransformMatrix3D::Identity();
        TransformMatrix3D world_to_camera_guess = TransformMatrix3D::Identity();

        bool _has_guess = false;

        ImageFrame()
        {
            robot_to_world.setIdentity();
        }

        ~ImageFrame()
        {
            if (d_keypoints_pos)
                checkCudaErrors(cudaFree(d_keypoints_pos));
            if (d_keypoints_score)
                checkCudaErrors(cudaFree(d_keypoints_score));
            if (d_descriptors1)
                checkCudaErrors(cudaFree(d_descriptors1));
            if (d_descriptors2)
                checkCudaErrors(cudaFree(d_descriptors2));
            if (d_keypoints_num)
                checkCudaErrors(cudaFree(d_keypoints_num));
            if (d_points)
                checkCudaErrors(cudaFree(d_points));
            if (d_matched_landmarks)
                checkCudaErrors(cudaFree(d_matched_landmarks));
            if (d_matched_points)
                checkCudaErrors(cudaFree(d_matched_points));
            if (d_num_of_matched_landmarks)
                checkCudaErrors(cudaFree(d_num_of_matched_landmarks));
            if (d_num_of_matched_points)
                checkCudaErrors(cudaFree(d_num_of_matched_points));
        }

    } ImageFrame_t;

    typedef std::shared_ptr<Jetracer::ImageFrame> pImageFrame;

    typedef struct RgbdFrame : ImageFrame
    {

        // image is used for sending it over to foxglove
        // usually, result of processing
        unsigned char *h_image;
        size_t h_image_length;

        unsigned char *h_rgb_image;
        uint16_t *h_depth_image;

        bool depth_aligned = false;
        std::function<void(pBaseFrame rgbd_frame, TmpData_t &tmp_frame)> depthFrameAligner;

        size_t rgb_image_size;
        size_t depth_image_size;

        unsigned char *d_rgb_image;
        unsigned char *d_grayscale_image;
        uint16_t *d_depth_image;
        uint16_t *d_depth_image_aligned;

        std::size_t rgb_pitch;
        std::size_t grayscale_pitch;

        int2 rgb_image_resolution;
        int2 depth_image_resolution;

        int rgb_frame_id;
        int depth_frame_id;

        float2 *d_features_image_positions;
        uint32_t *d_features_descriptors;

        int h_features_count;

        void uploadToGPU(cudaStream_t stream)
        {
            if (!d_rgb_image)
            {
                checkCudaErrors(cudaMallocPitch((void **)&d_rgb_image,
                                                &rgb_pitch, rgb_image_resolution.x * sizeof(char),
                                                rgb_image_resolution.y * sizeof(char) * 3));
            };

            checkCudaErrors(cudaMemcpy2DAsync((void *)d_rgb_image,
                                              rgb_pitch,
                                              h_rgb_image,
                                              rgb_image_resolution.x,
                                              rgb_image_resolution.x,
                                              rgb_image_resolution.y * 3,
                                              cudaMemcpyHostToDevice,
                                              stream));
            if (!d_grayscale_image)
            {
                checkCudaErrors(cudaMallocPitch((void **)&d_grayscale_image,
                                                &grayscale_pitch, rgb_image_resolution.x * sizeof(char),
                                                rgb_image_resolution.y * sizeof(char)));
            };

            if (!d_depth_image_aligned)
            {
                checkCudaErrors(cudaMalloc((void **)&d_depth_image_aligned,
                                           depth_image_size));
            };

            if (depth_aligned)
            {

                checkCudaErrors(cudaMemcpyAsync((void *)d_depth_image_aligned,
                                                h_depth_image,
                                                depth_image_size,
                                                cudaMemcpyHostToDevice,
                                                stream));
            }
            else
            {
                if (!d_depth_image)
                {
                    checkCudaErrors(cudaMalloc((void **)&d_depth_image,
                                               depth_image_size));
                };

                checkCudaErrors(cudaMemcpyAsync((void *)d_depth_image,
                                                h_depth_image,
                                                depth_image_size,
                                                cudaMemcpyHostToDevice,
                                                stream));
            }
        }

        ~RgbdFrame()
        {
            if (h_image)
                free(h_image);
            if (h_rgb_image)
                free(h_rgb_image);
            if (h_depth_image)
                free(h_depth_image);
            if (d_rgb_image)
                checkCudaErrors(cudaFree(d_rgb_image));
            if (d_depth_image)
                checkCudaErrors(cudaFree(d_depth_image));
            if (d_depth_image_aligned)
                checkCudaErrors(cudaFree(d_depth_image_aligned));
            if (d_features_image_positions)
                checkCudaErrors(cudaFree(d_features_image_positions));
            if (d_features_descriptors)
                checkCudaErrors(cudaFree(d_features_descriptors));
            if (d_grayscale_image)
                checkCudaErrors(cudaFree(d_grayscale_image));
            // if ()
            //     checkCudaErrors(cudaFree());
        }
    } RgbdFrame_t;

    typedef std::shared_ptr<Jetracer::RgbdFrame> pRgbdFrame;

    typedef struct ImuFrame : BaseFrame
    {
        rs2_vector motion_data;
    } ImuFrame_t;

}

#endif // JETRACER_FRAME_TYPES_H
