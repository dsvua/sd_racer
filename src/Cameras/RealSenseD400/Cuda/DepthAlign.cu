// #include "jetracer_rscuda_utils.cuh"
#include "Cameras/RealSenseD400/Cuda/DepthAlign.h"
#include "Cameras/RealSenseD400/RealSenseD400.h"
#include "Cuda/CudaCommon.h"
#include <iostream>
#include <stdio.h> //for printf

#define RS2_CUDA_THREADS_PER_BLOCK 32

namespace Jetracer
{
    __global__ void kernel_depth_to_other(uint16_t *aligned_out,
                                          const uint16_t *depth_in,
                                          const int2 *mapped_pixels,
                                          const rs2_intrinsics *depth_intrin,
                                          const rs2_intrinsics *other_intrin)
    {
        int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
        int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

        auto depth_size = depth_intrin->width * depth_intrin->height;
        int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

        if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
            return;

        int2 p0 = mapped_pixels[depth_pixel_index];
        int2 p1 = mapped_pixels[depth_size + depth_pixel_index];

        if (p0.x < 0 || p0.y < 0 || p1.x >= other_intrin->width || p1.y >= other_intrin->height)
            return;

        // Transfer between the depth pixels and the pixels inside the rectangle on the other image
        unsigned int new_val = depth_in[depth_pixel_index];
        unsigned int *arr = (unsigned int *)aligned_out;
        for (int y = p0.y; y <= p1.y; ++y)
        {
            for (int x = p0.x; x <= p1.x; ++x)
            {
                auto other_pixel_index = y * other_intrin->width + x;
                new_val = new_val << 16 | new_val;
                atomicMin(&arr[other_pixel_index / 2], new_val);
            }
        }
    }

    __global__ void kernel_replace_to_zero(uint16_t *aligned_out,
                                           const rs2_intrinsics *other_intrin)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < other_intrin->width && y < other_intrin->height)

        {
            auto other_pixel_index = y * other_intrin->width + x;
            if (aligned_out[other_pixel_index] == 0xffff)
                aligned_out[other_pixel_index] = 0;
        }
    }

    /* Given pixel coordinates and depth in an image with no distortion or inverse distortion coefficients, compute the corresponding point in 3D space relative to the same camera */
    __device__ static void rs2_deproject_pixel_to_point(float point[3],
                                                        const struct rs2_intrinsics *intrin,
                                                        const float pixel[2],
                                                        float depth)
    {
        assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
        assert(intrin->model != RS2_DISTORTION_FTHETA);                 // Cannot deproject to an ftheta image
        // assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model

        float x = (pixel[0] - intrin->ppx) / intrin->fx;
        float y = (pixel[1] - intrin->ppy) / intrin->fy;

        if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
        {
            float r2 = x * x + y * y;
            float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
            float ux = x * f + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
            float uy = y * f + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
            x = ux;
            y = uy;
        }
        point[0] = depth * x;
        point[1] = depth * y;
        point[2] = depth;
    }

    /* Given a point in 3D space, compute the corresponding pixel coordinates in an image with no distortion or forward distortion coefficients produced by the same camera */
    __device__ static void rs2_project_point_to_pixel(float pixel[2],
                                                      const struct rs2_intrinsics *intrin,
                                                      const float point[3])
    {
        // assert(intrin->model != RS2_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image

        float x = point[0] / point[2], y = point[1] / point[2];

        if (intrin->model == RS2_DISTORTION_MODIFIED_BROWN_CONRADY)
        {

            float r2 = x * x + y * y;
            float f = 1 + intrin->coeffs[0] * r2 + intrin->coeffs[1] * r2 * r2 + intrin->coeffs[4] * r2 * r2 * r2;
            x *= f;
            y *= f;
            float dx = x + 2 * intrin->coeffs[2] * x * y + intrin->coeffs[3] * (r2 + 2 * x * x);
            float dy = y + 2 * intrin->coeffs[3] * x * y + intrin->coeffs[2] * (r2 + 2 * y * y);
            x = dx;
            y = dy;
        }

        if (intrin->model == RS2_DISTORTION_FTHETA)
        {
            float r = sqrtf(x * x + y * y);
            float rd = (float)(1.0f / intrin->coeffs[0] * atan(2 * r * tan(intrin->coeffs[0] / 2.0f)));
            x *= rd / r;
            y *= rd / r;
        }

        pixel[0] = x * intrin->fx + intrin->ppx;
        pixel[1] = y * intrin->fy + intrin->ppy;
    }

    /* Transform 3D coordinates relative to one sensor to 3D coordinates relative to another viewpoint */
    __device__ static void rs2_transform_point_to_point(float to_point[3],
                                                        const struct rs2_extrinsics *extrin,
                                                        const float from_point[3])
    {
        to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] + extrin->rotation[6] * from_point[2] + extrin->translation[0];
        to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[7] * from_point[2] + extrin->translation[1];
        to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
    }

    __device__ void kernel_transfer_pixels(int2 *mapped_pixels,
                                           const rs2_intrinsics *depth_intrin,
                                           const rs2_intrinsics *other_intrin,
                                           const rs2_extrinsics *depth_to_other,
                                           float depth_val,
                                           int depth_x,
                                           int depth_y,
                                           int block_index)
    {
        float shift = block_index ? 0.5 : -0.5;
        auto depth_size = depth_intrin->width * depth_intrin->height;
        auto mapped_index = block_index * depth_size + (depth_y * depth_intrin->width + depth_x);

        if (mapped_index >= depth_size * 2)
            return;

        // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
        if (depth_val == 0)
        {
            mapped_pixels[mapped_index] = {-1, -1};
            return;
        }

        //// Map the top-left corner of the depth pixel onto the other image
        float depth_pixel[2] = {depth_x + shift, depth_y + shift}, depth_point[3], other_point[3], other_pixel[2];
        Jetracer::rs2_deproject_pixel_to_point(depth_point, depth_intrin, depth_pixel, depth_val);
        Jetracer::rs2_transform_point_to_point(other_point, depth_to_other, depth_point);
        Jetracer::rs2_project_point_to_pixel(other_pixel, other_intrin, other_point);
        mapped_pixels[mapped_index].x = static_cast<int>(other_pixel[0] + 0.5f);
        mapped_pixels[mapped_index].y = static_cast<int>(other_pixel[1] + 0.5f);
    }

    __global__ void kernel_map_depth_to_other(int2 *mapped_pixels,
                                              const uint16_t *depth_in,
                                              const rs2_intrinsics *depth_intrin,
                                              const rs2_intrinsics *other_intrin,
                                              const rs2_extrinsics *depth_to_other,
                                              float depth_scale)
    {
        int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
        int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

        int depth_pixel_index = depth_y * depth_intrin->width + depth_x;
        if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
            return;
        float depth_val = depth_in[depth_pixel_index] * depth_scale;
        kernel_transfer_pixels(mapped_pixels, depth_intrin, other_intrin, depth_to_other, depth_val, depth_x, depth_y, blockIdx.z);
    }

    void depthAligner(pBaseFrame rgbd_frame,
                      TmpData_t &tmp_frame)
    {
        // std::cout << "depthAligner" << std::endl;
        pRealSenseD400RgbdFrame realsense_rgbd_frame = std::static_pointer_cast<RealSenseD400RgbdFrame>(rgbd_frame);

        if (!tmp_frame.d_depth_intrinsics)
            tmp_frame.d_depth_intrinsics = make_device_copy(realsense_rgbd_frame->depth_intrinsics, tmp_frame.stream);
        if (!tmp_frame.d_rgb_intrinsics)
            tmp_frame.d_rgb_intrinsics = make_device_copy(realsense_rgbd_frame->rgb_intrinsics, tmp_frame.stream);
        if (!tmp_frame.d_depth_other_extrinsics)
            tmp_frame.d_depth_other_extrinsics = make_device_copy(realsense_rgbd_frame->extrinsics, tmp_frame.stream);

        int depth_pixel_count = realsense_rgbd_frame->depth_intrinsics.width * realsense_rgbd_frame->depth_intrinsics.height;
        int other_pixel_count = realsense_rgbd_frame->rgb_intrinsics.width * realsense_rgbd_frame->rgb_intrinsics.height;
        int aligned_pixel_count = other_pixel_count;

        int aligned_byte_size = aligned_pixel_count * 2;

        if (!tmp_frame.d_pixel_map)
            tmp_frame.d_pixel_map = alloc_dev<int2>(depth_pixel_count * 2);

        cudaMemset(realsense_rgbd_frame->d_depth_image_aligned, 0xff, aligned_byte_size);

        // config threads
        dim3 threads(RS2_CUDA_THREADS_PER_BLOCK, RS2_CUDA_THREADS_PER_BLOCK);
        dim3 depth_blocks(calc_block_size(realsense_rgbd_frame->depth_intrinsics.width, threads.x),
                          calc_block_size(realsense_rgbd_frame->depth_intrinsics.height, threads.y));
        dim3 other_blocks(calc_block_size(realsense_rgbd_frame->rgb_intrinsics.width, threads.x),
                          calc_block_size(realsense_rgbd_frame->rgb_intrinsics.height, threads.y));
        dim3 mapping_blocks(depth_blocks.x, depth_blocks.y, 2);

        kernel_map_depth_to_other<<<mapping_blocks, threads, 0, tmp_frame.stream>>>(tmp_frame.d_pixel_map.get(),
                                                                                    realsense_rgbd_frame->d_depth_image,
                                                                                    tmp_frame.d_depth_intrinsics.get(),
                                                                                    tmp_frame.d_rgb_intrinsics.get(),
                                                                                    tmp_frame.d_depth_other_extrinsics.get(),
                                                                                    realsense_rgbd_frame->depth_scale);
        // checkCudaErrors(cudaStreamSynchronize(tmp_frame.stream));

        kernel_depth_to_other<<<depth_blocks, threads, 0, tmp_frame.stream>>>(realsense_rgbd_frame->d_depth_image_aligned,
                                                                              realsense_rgbd_frame->d_depth_image,
                                                                              tmp_frame.d_pixel_map.get(),
                                                                              tmp_frame.d_depth_intrinsics.get(),
                                                                              tmp_frame.d_rgb_intrinsics.get());
        // checkCudaErrors(cudaStreamSynchronize(tmp_frame.stream));

        kernel_replace_to_zero<<<other_blocks, threads, 0, tmp_frame.stream>>>(realsense_rgbd_frame->d_depth_image_aligned,
                                                                               tmp_frame.d_rgb_intrinsics.get());
        // checkCudaErrors(cudaStreamSynchronize(tmp_frame.stream));
    }

}
