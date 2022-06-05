#ifndef JETRACER_FRAME_TYPES_H
#define JETRACER_FRAME_TYPES_H

#include <cuda_runtime.h>
#include <helper_cuda.h>
// #include <eigen3/Eigen/Eigen>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

namespace Jetracer
{

#pragma once

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

        int max_keypoints_num = 0;

        pBaseFrame previous_frame;

    } TmpData_t;

    typedef struct ImageFrame : BaseFrame
    {
        FrameType frame_type;
        double timestamp;
        bool keypoints_detected = false;

        // For ORB
        float2 *d_keypoints_pos;
        float *d_keypoints_score;
        unsigned char *d_descriptors;

        ~ImageFrame()
        {
            if (d_keypoints_pos)
                checkCudaErrors(cudaFree(d_keypoints_pos));
            if (d_keypoints_score)
                checkCudaErrors(cudaFree(d_keypoints_score));
            if (d_descriptors)
                checkCudaErrors(cudaFree(d_descriptors));
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
