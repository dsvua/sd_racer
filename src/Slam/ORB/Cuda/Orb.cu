#include "Slam/ORB/Cuda/Orb.h"
#include "Cuda/CudaCommon.h"

#include <vector>
#include <helper_cuda.h>
#include <cstdio> // for printf

// #include <cooperative_groups.h>
// using namespace cooperative_groups;

namespace Jetracer
{

    __constant__ unsigned char c_pattern[sizeof(int2) * 512];

#define GET_VALUE(idx)                                                                      \
    image[(loc.y + __float2int_rn(pattern[idx].x * b + pattern[idx].y * a)) * image_pitch + \
          loc.x + __float2int_rn(pattern[idx].x * a - pattern[idx].y * b)]

    // __global__ void calcOrb_kernel(const PtrStepb image, float2 *d_keypoints_pos, const int npoints, PtrStepb descriptors)
    __global__ void calc_orb_kernel(float *d_keypoints_angle,
                                    float2 *d_keypoints_pos,
                                    unsigned char *d_descriptors,
                                    unsigned char *image,
                                    int image_pitch,
                                    int image_width,
                                    int image_height,
                                    int keypoints_num)
    {
        int id = blockIdx.x;
        int tid = threadIdx.x;
        if (id >= keypoints_num)
            return;

        const float2 kpt = d_keypoints_pos[id];
        short2 loc = make_short2(short(kpt.x), short(kpt.y));
        unsigned char *desc = d_descriptors;
        if (loc.x < 17 || loc.x > image_width - 17 || loc.y < 17 || loc.y > image_height - 17)
        {
            desc[id * 32 + tid] = 0;
            return;
        }

        const int2 *pattern = ((int2 *)c_pattern) + 16 * tid;

        const float factorPI = (float)(CUDART_PI_F / 180.f);
        float angle = d_keypoints_angle[id] * factorPI;

        float a = (float)cosf(angle);
        float b = (float)sinf(angle);

        int t0, t1, val;
        t0 = GET_VALUE(0);
        t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2);
        t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[id * 32 + tid] = (unsigned char)val;
    }

    __global__ void compute_fast_angle_kernel(float *d_keypoints_angle,
                                              float2 *d_keypoints_pos,
                                              unsigned char *image,
                                              int image_pitch,
                                              int image_width,
                                              int image_height,
                                              int keypoints_num)
    {
        int idx = blockIdx.x;
        int k_x = floorf(d_keypoints_pos[idx].x + 0.5);
        int k_y = floorf(d_keypoints_pos[idx].y + 0.5);
        // Hardcoding for patch 31x31, so, radius is 15
        int r2 = 15 * 15;
        float m10 = 0;
        float m01 = 0;

        // Hardcoding for patch 31x31, so, radius is 15
        if (threadIdx.x < 31)
        {
            int mult_dx = threadIdx.x - 15;
            int tdx = threadIdx.x + k_x - 15;
            if (tdx > 0 && tdx < image_width)
            {
                m10 = mult_dx * image[k_y * image_pitch + tdx];
            }
        }

        for (int dy = 1; dy < 16; dy++)
        {
            int dx = floor(sqrtf(r2 - float(dy * dy)) + 0.5);
            if (threadIdx.x > 14 - dx && threadIdx.x < 16 + dx)
            {
                int mult_dx = threadIdx.x - 15;
                int tdx = k_x + threadIdx.x - 15;

                if (k_y - dy > 0 && tdx > 0 && tdx < image_width)
                {
                    float i = image[(k_y - dy) * image_pitch + tdx];
                    m01 -= dy * i;
                    m10 += mult_dx * i;
                }

                if (k_y + dy < image_height && tdx > 0 && tdx < image_width)
                {
                    float i = image[(k_y + dy) * image_pitch + tdx];
                    m01 += dy * i;
                    m10 += mult_dx * i;
                }
            }
        }

        __syncthreads();

        for (int offset = 16; offset > 0; offset /= 2)
        {
            m01 += __shfl_down_sync(FULL_MASK, m01, offset);
            m10 += __shfl_down_sync(FULL_MASK, m10, offset);
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            d_keypoints_angle[idx] = atan2f(m01, m10);
        }
    }

    // __device__ int atomicAggInc(int *ctr)
    // {
    //     auto g = coalesced_threads();
    //     int warp_res;
    //     if (g.thread_rank() == 0)
    //         warp_res = atomicAdd(ctr, g.size());
    //     return g.shfl(warp_res, 0) + g.thread_rank();
    // }

    __global__ void filter_keypoints_kernel(float *src_keypoints_score,
                                            float2 *src_keypoints_pos,
                                            unsigned char *src_descriptors,
                                            float *dst_keypoints_score,
                                            float2 *dst_keypoints_pos,
                                            unsigned char *dst_descriptors,
                                            int max_keypoints_num,
                                            unsigned int *keypoints_num,
                                            float min_score)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < max_keypoints_num)
        {
            float score = src_keypoints_score[idx];
            float2 pos = src_keypoints_pos[idx];

            if (score > min_score)
            {
                // int newIdx = atomicAggInc(keypoints_num);
                int newIdx = atomicAdd(keypoints_num, 1);
                dst_keypoints_score[newIdx] = score;
                dst_keypoints_pos[newIdx] = pos;
                for (int i = 0; i < 32; i++)
                {
                    dst_descriptors[newIdx * 32 + i] = src_descriptors[idx * 32 + i];
                }
            }
        }
    }

    void compute_fast_angle(pRgbdFrame current_frame, TmpData_t &tmp_frame)
    {
        // sdfs
        compute_fast_angle_kernel<<<tmp_frame.max_keypoints_num, 32, 0, tmp_frame.stream>>>(tmp_frame.d_keypoints_angle,
                                                                                            tmp_frame.d_keypoints_pos,
                                                                                            current_frame->d_grayscale_image,
                                                                                            current_frame->grayscale_pitch,
                                                                                            current_frame->rgb_image_resolution.x,
                                                                                            current_frame->rgb_image_resolution.y,
                                                                                            tmp_frame.max_keypoints_num);
        // CUDA_KERNEL_CHECK();
    }

    void calc_orb(pRgbdFrame current_frame, TmpData_t &tmp_frame)
    {
        calc_orb_kernel<<<tmp_frame.max_keypoints_num, CUDA_WARP_SIZE, 0, tmp_frame.stream>>>(tmp_frame.d_keypoints_angle,
                                                                                              tmp_frame.d_keypoints_pos,
                                                                                              tmp_frame.d_descriptors,
                                                                                              current_frame->d_grayscale_image,
                                                                                              current_frame->grayscale_pitch,
                                                                                              current_frame->rgb_image_resolution.x,
                                                                                              current_frame->rgb_image_resolution.y,
                                                                                              tmp_frame.max_keypoints_num);
        // CUDA_KERNEL_CHECK();
    }

    void filter_keypoints(pRgbdFrame current_frame, TmpData_t &tmp_frame, float min_score)
    {

        checkCudaErrors(cudaMemcpyAsync(current_frame->d_keypoints_num,
                                        &current_frame->keypoints_num,
                                        sizeof(unsigned int),
                                        cudaMemcpyHostToDevice,
                                        tmp_frame.stream));

        dim3 threads(CUDA_WARP_SIZE);
        dim3 blocks(calc_block_size(tmp_frame.max_keypoints_num, threads.x));

        filter_keypoints_kernel<<<blocks, threads, 0, tmp_frame.stream>>>(tmp_frame.d_keypoints_score,
                                                                          tmp_frame.d_keypoints_pos,
                                                                          tmp_frame.d_descriptors,
                                                                          current_frame->d_keypoints_score,
                                                                          current_frame->d_keypoints_pos,
                                                                          current_frame->d_descriptors,
                                                                          tmp_frame.max_keypoints_num,
                                                                          current_frame->d_keypoints_num,
                                                                          min_score);
        checkCudaErrors(cudaMemcpyAsync(&current_frame->keypoints_num,
                                        current_frame->d_keypoints_num,
                                        sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost,
                                        tmp_frame.stream));

        // CUDA_KERNEL_CHECK();
    }

    void loadPattern()
    {
        const int npoints = 512;
        std::vector<int2> pattern;
        const int2 *pattern0 = (const int2 *)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        checkCudaErrors(cudaMemcpyToSymbol(c_pattern, pattern.data(), sizeof(int2) * 512));
    }
} // namespace Jetracer
