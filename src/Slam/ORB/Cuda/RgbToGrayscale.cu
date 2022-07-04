#include "Slam/ORB/Cuda/RgbToGrayscale.h"
#include "Types/Defines.h"
#include "Cuda/CudaCommon.h"
#include <iostream>

#define RS2_CUDA_THREADS_PER_BLOCK 32

namespace Jetracer
{
    __global__ void kernel_rgb_to_grayscale(unsigned char *dst, unsigned char *src, int cols, int rows, int dst_pitch, int src_pitch)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < cols && y < rows)
        {
            float R, G, B;
            R = float(src[y * src_pitch + x * 3 + 0]);
            G = float(src[y * src_pitch + x * 3 + 1]);
            B = float(src[y * src_pitch + x * 3 + 2]);
            dst[y * dst_pitch + x] = floor((B * 0.07 + G * 0.72 + R * 0.21) + 0.5);
        }
    }

    // Just for testing Eigen matrixes use in kernels
    __global__ void printing_eigen_kernel(PointCoordinates *d_testvector)
    {
        PointCoordinates d_testvector2 = *d_testvector;
        printf("d_testvector2: ");
        for (int i=0; i < 3; i++)
            printf("\t %0.2f", d_testvector2(i,1));
        printf("\n");
    }

    // void printing_eigen(TmpData_t &tmp_frame)
    // {
    //     PointCoordinates* d_testvector;
    //     PointCoordinates h_testvector;
    //     h_testvector << 3, 2, 1;
    //     std::cout << "h_testvector: " << h_testvector << std::endl;

    //     checkCudaErrors(cudaMalloc((void **)&d_testvector, sizeof(PointCoordinates)));
    //     checkCudaErrors(cudaMemcpyAsync((void *)d_testvector,
    //                                     (void *)&h_testvector,
    //                                     sizeof(PointCoordinates),
    //                                     cudaMemcpyHostToDevice,
    //                                     tmp_frame.stream));

    //     printing_eigen<<<1,1,0,tmp_frame.stream>>>(d_testvector);

    // }

    void rgb_to_grayscale(pRgbdFrame current_frame, TmpData_t &tmp_frame)
    {

        dim3 threads(RS2_CUDA_THREADS_PER_BLOCK, RS2_CUDA_THREADS_PER_BLOCK);
        dim3 blocks(calc_block_size(current_frame->rgb_image_resolution.x, threads.x), calc_block_size(current_frame->rgb_image_resolution.y, threads.y));

        kernel_rgb_to_grayscale<<<blocks, threads, 0, tmp_frame.stream>>>(current_frame->d_grayscale_image,
                                                                          current_frame->d_rgb_image,
                                                                          current_frame->rgb_image_resolution.x,
                                                                          current_frame->rgb_image_resolution.y,
                                                                          current_frame->grayscale_pitch,
                                                                          current_frame->rgb_pitch);
        
        // CUDA_KERNEL_CHECK();
    }
}
