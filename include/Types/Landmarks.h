#ifndef JETRACER_CUDA_LANDMARKS_H
#define JETRACER_CUDA_LANDMARKS_H

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Defines.h"
#include "Math.h"

// should be devideble by 32
#define STARTING_NUM_OF_MEASUREMENTS 1024
#define STARTING_NUM_OF_LANDMARKS 102400

namespace Jetracer
{
    typedef struct Measurement
    {
        Vector3f camera_coordinates;
        float inverse_depth_meters;

    } Measurement_t;

    typedef struct Landmark
    {
        // ds world coordinates of the landmark
        Vector3f world_coordinates;

        float inverse_depth_meters;
        int num_of_measurements = 0;
        Measurement *measurements;

    } Landmark_t;

    typedef struct Landmarks
    {
        Landmark_t *data;
        Landmark_t *tmp_data;
        int size = 0;
        int max_size = STARTING_NUM_OF_LANDMARKS;

        Landmarks()
        {
            checkCudaErrors(cudaMalloc((void **)&data,
                                       sizeof(Landmark) * STARTING_NUM_OF_LANDMARKS));
            fillMeasurements(0, max_size);
        }

        ~Landmarks()
        {
            if (data)
                checkCudaErrors(cudaFree(data));
        }

        void expand(cudaStream_t stream)
        {
            expand(max_size, stream);
        }

        void expand(int num_of_additional_landmarks, cudaStream_t stream)
        {
            tmp_data = data;
            checkCudaErrors(cudaMallocAsync((void **)&data,
                                            sizeof(Landmark) * (max_size + num_of_additional_landmarks),
                                            stream));
            checkCudaErrors(cudaMemcpyAsync((void **)&data,
                                            (void **)&tmp_data,
                                            sizeof(Landmark) * max_size,
                                            cudaMemcpyDeviceToDevice,
                                            stream));
            checkCudaErrors(cudaFreeAsync(tmp_data, stream));

            fillMeasurements(max_size, num_of_additional_landmarks, stream);

            max_size += num_of_additional_landmarks;
        }

        void fillMeasurements(int start, int num, cudaStream_t stream = nullptr)
        {
            Landmark_t landmark;

            for (int i = start; i < start + num; i++)
            {

                if (stream)
                {
                    checkCudaErrors(cudaMallocAsync((void **)&landmark.measurements,
                                                    sizeof(Measurement) * STARTING_NUM_OF_MEASUREMENTS,
                                                    stream));
                    checkCudaErrors(cudaMemcpyAsync((void **)&data[i],
                                                    (void **)&landmark,
                                                    sizeof(Landmark),
                                                    cudaMemcpyHostToDevice,
                                                    stream));
                }
                else
                {
                    checkCudaErrors(cudaMalloc((void **)&landmark.measurements,
                                               sizeof(Measurement) * STARTING_NUM_OF_MEASUREMENTS));
                    checkCudaErrors(cudaMemcpy((void **)&data[i],
                                               (void **)&landmark,
                                               sizeof(Landmark),
                                               cudaMemcpyHostToDevice));
                }
            }
        }

    } Landmarks_t;

}

#endif // JETRACER_CUDA_LANDMARKS_H
