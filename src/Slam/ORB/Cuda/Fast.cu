#include "Slam/ORB/Cuda/Fast.h"
#include "Slam/ORB/Cuda/Nms.h"
#include "Cuda/CudaCommon.h"

namespace Jetracer
{
    // ---------------------------------------
    //                kernels
    // ---------------------------------------
    __inline__ __device__ unsigned char fast_gpu_is_corner(const unsigned int &address,
                                                           const int &min_arc_length)
    {
        int ones = __popc(address);
        if (ones < min_arc_length)
        { // if we dont have enough 1-s in the address, dont even try
            return 0;
        }
        unsigned int address_dup = address | (address << 16); // duplicate the low 16-bits at the high 16-bits
        while (ones > 0)
        {
            address_dup <<= __clz(address_dup); // shift out the high order zeros
            int lones = __clz(~address_dup);    // count the leading ones
            if (lones >= min_arc_length)
            {
                return 1;
            }
            address_dup <<= lones; // shift out the high order ones
            ones -= lones;
        }
        return 0;
    }

    __global__ void fast_gpu_calculate_lut_kernel(unsigned char *__restrict__ d_corner_lut,
                                                  const int min_arc_length)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x; // all 16 bits come from here
        d_corner_lut[x] = fast_gpu_is_corner(x, min_arc_length);
    }

    __inline__ __device__ int bresenham_circle_offset_pitch(const int &i,
                                                            const int &pitch,
                                                            const int &pitch2,
                                                            const int &pitch3)
    {
        /*
         * Note to future self and others:
         * this function is only should be called in for loops that are unrolled.
         * Due to unrollment, the if else structure disappears, and the offsets get
         * substituted.
         *
         * Order within the circle:
         *
         *      7 8  9
         *    6       10
         *  5           11
         *  4     x     12
         *  3           13
         *   2        14
         *     1 0 15
         */
        int offs = 0;
        if (i == 0)
            offs = pitch3;
        else if (i == 1)
            offs = pitch3 - 1;
        else if (i == 2)
            offs = pitch2 - 2;
        else if (i == 3)
            offs = pitch - 3;
        else if (i == 4)
            offs = -3;
        else if (i == 5)
            offs = -pitch - 3;
        else if (i == 6)
            offs = -pitch2 - 2;
        else if (i == 7)
            offs = -pitch3 - 1;
        else if (i == 8)
            offs = -pitch3;
        else if (i == 9)
            offs = -pitch3 + 1;
        else if (i == 10)
            offs = -pitch2 + 2;
        else if (i == 11)
            offs = -pitch + 3;
        else if (i == 12)
            offs = 3;
        else if (i == 13)
            offs = pitch + 3;
        else if (i == 14)
            offs = pitch2 + 2;
        else if (i == 15)
            offs = pitch3 + 1;
        return offs;
    }

    __inline __device__ unsigned int fast_gpu_prechecks(const float &c_t,
                                                        const float &ct,
                                                        const unsigned char *image_ptr,
                                                        const int &image_pitch,
                                                        const int &image_pitch2,
                                                        const int &image_pitch3)
    {
        /*
         * Note to future self:
         * using too many prechecks of course doesnt help
         */
        // (-3,0) (3,0) -> 4,12
        float px0 = (float)image_ptr[bresenham_circle_offset_pitch(4, image_pitch, image_pitch2, image_pitch3)];
        float px1 = (float)image_ptr[bresenham_circle_offset_pitch(12, image_pitch, image_pitch2, image_pitch3)];
        if ((signbit(px0 - c_t) | signbit(px1 - c_t) | signbit(ct - px0) | signbit(ct - px1)) == 0)
        {
            return 1;
        }
        // (0,3), (0,-3) -> 0, 8
        px0 = (float)image_ptr[bresenham_circle_offset_pitch(0, image_pitch, image_pitch2, image_pitch3)];
        px1 = (float)image_ptr[bresenham_circle_offset_pitch(8, image_pitch, image_pitch2, image_pitch3)];
        if ((signbit(px0 - c_t) | signbit(px1 - c_t) | signbit(ct - px0) | signbit(ct - px1)) == 0)
        {
            return 1;
        }
        return 0;
    }

    __inline__ __device__ int fast_gpu_is_corner_quick(
        const unsigned char *__restrict__ d_corner_lut,
        const float *__restrict__ px,
        const float &center_value,
        const float &threshold,
        unsigned int &dark_diff_address,
        unsigned int &bright_diff_address)
    {
        const float ct = center_value + threshold;
        const float c_t = center_value - threshold;
        dark_diff_address = 0;
        bright_diff_address = 0;

#pragma unroll 16
        for (int i = 0; i < 16; ++i)
        {
            int darker = signbit(px[i] - c_t);
            int brighter = signbit(ct - px[i]);
            dark_diff_address += signbit(px[i] - c_t) ? (1 << i) : 0;
            bright_diff_address += signbit(ct - px[i]) ? (1 << i) : 0;
        }
        return (d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address]);
    }

    __global__ void fast_gpu_calc_corner_response_kernel(
        const int image_width,
        const int image_height,
        const int image_pitch,
        const unsigned char *__restrict__ d_image,
        const int horizontal_border,
        const int vertical_border,
        const unsigned char *__restrict__ d_corner_lut,
        const float threshold,
        const int min_arc_length,
        const int response_pitch_elements,
        float *__restrict__ d_response)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x; // thread id X
        const int y = blockDim.y * blockIdx.y + threadIdx.y; // thread id Y
        if (x < image_width && y < image_height)
        {
            const int resp_offset = y * response_pitch_elements + x;
            d_response[resp_offset] = 0.0f;
            if ((x >= horizontal_border) &&
                (y >= vertical_border) &&
                (x < (image_width - horizontal_border)) &&
                (y < (image_height - vertical_border)))
            {
                const unsigned char *d_image_ptr = d_image + y * image_pitch + x;
                const float c = (float)(*d_image_ptr);
                const float ct = c + threshold;
                const float c_t = c - threshold;
                /*
                 * Note to future self:
                 * we need to create 2 differences for each of the 16 pixels
                 * have 1 lookup table, and look-up both values
                 *
                 * c_t stands for: c - threshold (epsilon)
                 * ct stands for : c + threshold (epsilon)
                 *
                 * Label of px:
                 * - darker  if   px < c_t              (1)
                 * - similar if   c_t <= px <= ct      (2)
                 * - brighter if  ct < px             (3)
                 *
                 * Darker diff: px - c_t
                 * sign will only give 1 in case of (1), and 0 in case of (2),(3)
                 *
                 * Similarly, brighter diff: ct - px
                 * sign will only give 1 in case of (3), and 0 in case of (2),(3)
                 */
                unsigned int dark_diff_address = 0;
                unsigned int bright_diff_address = 0;

                // Precalculate pitches
                const int image_pitch2 = image_pitch << 1;
                const int image_pitch3 = image_pitch + (image_pitch << 1);

                // Do a coarse corner check
                // TODO: I could use the results of the prechecks afterwards
                if (fast_gpu_prechecks(c_t, ct, d_image_ptr, image_pitch, image_pitch2, image_pitch3))
                {
                    return;
                }

                float px[16];
#pragma unroll 16
                for (int i = 0; i < 16; ++i)
                {
                    int image_ptr_offset = bresenham_circle_offset_pitch(i, image_pitch, image_pitch2, image_pitch3);
                    px[i] = (float)d_image_ptr[image_ptr_offset];
                    int darker = signbit(px[i] - c_t);
                    int brighter = signbit(ct - px[i]);
                    dark_diff_address += signbit(px[i] - c_t) ? (1 << i) : 0;
                    bright_diff_address += signbit(ct - px[i]) ? (1 << i) : 0;
                }
                // Look up these addresses, whether they qualify for a corner
                // If any of these qualify for a corner, it is a corner candidate, yaay
                if (d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address])
                {
                    /*
                     * Note to future self:
                     * Only calculate the score once we determined that the pixel is considered
                     * a corner. This policy gave better results than computing the score
                     * for every pixel
                     */
                    float response_bright = 0.0f;
                    float response_dark = 0.0f;
#pragma unroll 16
                    for (int i = 0; i < 16; ++i)
                    {
                        float absdiff = fabsf(px[i] - c) - threshold;
                        response_dark += (dark_diff_address & (1 << i)) ? absdiff : 0.0f;
                        response_bright += (bright_diff_address & (1 << i)) ? absdiff : 0.0f;
                    }
                    d_response[resp_offset] = fmaxf(response_bright, response_dark);
                }
            }
        }
    }

    // ---------------------------------------
    //            host functions
    // ---------------------------------------
    void fast_gpu_calculate_lut(unsigned char *d_corner_lut,
                                const int &min_arc_length)
    {
        // every thread writes a byte: in total 64kB gets written
        kernel_params_t p = cuda_gen_kernel_params_1d(64 * 1024, 256);
        dim3 threads(256);
        dim3 blocks((64 * 1024 + threads.x - 1) / threads.x);
        fast_gpu_calculate_lut_kernel<<<p.blocks_per_grid, p.threads_per_block>>>(d_corner_lut,
                                                                                  min_arc_length);
        // checkCudaErrors(cudaStreamSynchronize(stream));
    }

    void fast_gpu_calc_corner_response(const int image_width,
                                       const int image_height,
                                       const int image_pitch,
                                       const unsigned char *d_image,
                                       const int horizontal_border,
                                       const int vertical_border,
                                       const unsigned char *d_corner_lut,
                                       const float threshold,
                                       const int min_arc_length,
                                       const fast_score score,
                                       const int response_pitch_elements,
                                       float *d_response,
                                       cudaStream_t stream)
    {
        // Note: I'd like to launch 128 threads / thread block
        std::size_t threads_per_x = (image_width % CUDA_WARP_SIZE == 0) ? CUDA_WARP_SIZE : 16;
        std::size_t threads_per_y = 128 / threads_per_x;
        dim3 threads(threads_per_x, threads_per_y);
        dim3 blocks((image_width + threads.x - 1) / threads.x,
                    (image_height + threads.y - 1) / threads.y);

        fast_gpu_calc_corner_response_kernel<<<blocks, threads, 0, stream>>>(image_width,
                                                                             image_height,
                                                                             image_pitch,
                                                                             d_image,
                                                                             horizontal_border,
                                                                             vertical_border,
                                                                             d_corner_lut,
                                                                             threshold,
                                                                             min_arc_length,
                                                                             response_pitch_elements,
                                                                             d_response);
        // checkCudaErrors(cudaStreamSynchronize(stream));
    }

    void fast_detect(pRgbdFrame current_frame, TmpData_t &tmp_frame)
    {
        fast_gpu_calc_corner_response(current_frame->rgb_image_resolution.x,
                                      current_frame->rgb_image_resolution.y,
                                      current_frame->grayscale_pitch,
                                      current_frame->d_grayscale_image,
                                      3,
                                      3,
                                      tmp_frame.d_corner_lut,
                                      FAST_EPSILON,
                                      SUM_OF_ABS_DIFF_ON_ARC,
                                      FAST_SCORE,
                                      tmp_frame.d_keypoint_response_pitch / sizeof(float),
                                      tmp_frame.d_keypoint_response,
                                      tmp_frame.stream);

        grid_nms(current_frame, tmp_frame);
    }
} // namespace Jetracer
