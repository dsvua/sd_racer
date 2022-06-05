#ifndef JETRACER_CUDA_ALIGN_UTILS_H
#define JETRACER_CUDA_ALIGN_UTILS_H

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "../../../Types/Frames.h"
// #include "Slam/SlamLoop.h"

namespace Jetracer
{

    /** \brief Distortion model: defines how pixel coordinates should be mapped to sensor coordinates. */
    typedef enum rs2_distortion
    {
        RS2_DISTORTION_NONE,                   /**< Rectilinear images. No distortion compensation required. */
        RS2_DISTORTION_MODIFIED_BROWN_CONRADY, /**< Equivalent to Brown-Conrady distortion, except that tangential distortion is applied to radially distorted points */
        RS2_DISTORTION_INVERSE_BROWN_CONRADY,  /**< Equivalent to Brown-Conrady distortion, except undistorts image instead of distorting it */
        RS2_DISTORTION_FTHETA,                 /**< F-Theta fish-eye distortion model */
        RS2_DISTORTION_BROWN_CONRADY,          /**< Unmodified Brown-Conrady distortion model */
        RS2_DISTORTION_KANNALA_BRANDT4,        /**< Four parameter Kannala Brandt distortion model */
        RS2_DISTORTION_COUNT                   /**< Number of enumeration values. Not a valid input: intended to be used in for-loops. */
    } rs2_distortion;

    template <typename T>
    std::shared_ptr<T> alloc_dev(int elements)
    {
        T *d_data;
        auto res = cudaMalloc(&d_data, sizeof(T) * elements);
        if (res != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed status: " + res);
        return std::shared_ptr<T>(d_data, [](T *p)
                                  { cudaFree(p); });
    }

    template <typename T>
    std::shared_ptr<T> make_device_copy(T obj, cudaStream_t stream)
    {
        T *d_data;
        auto res = cudaMalloc(&d_data, sizeof(T));
        if (res != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed status: " + res);
        cudaMemcpyAsync(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice, stream);
        return std::shared_ptr<T>(d_data, [](T *data)
                                  { cudaFree(data); });
    }

    template <int N>
    struct bytes
    {
        unsigned char b[N];
    };

    const char *rs2_distortion_to_string(rs2_distortion distortion);

    void depthAligner(pBaseFrame rgbd_frame,
                      TmpData_t &tmp_frame);
}

#endif // JETRACER_CUDA_ALIGN_UTILS_H
