#ifndef JETRACER_CUDA_LANDMARKS_H
#define JETRACER_CUDA_LANDMARKS_H

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Defines.h"

// should be divisible by 32
#define MAX_NUM_OF_MEASUREMENTS 1024
#define STARTING_NUM_OF_LANDMARKS 32000

namespace Jetracer
{
    typedef struct Measurement
    {
        float inverse_depth_meters;
        PointCoordinates camera_coordinates;

    } Measurement_t;

    // structure of arrays
    typedef struct Landmarks
    {
        PointCoordinates *d_world_coordinates;
        PointCoordinates *d_camera_coordinates;
        float2 *d_camera_image_coordinates;
        uint *d_num_of_measurements;
        uint *d_global_id;
        uint4 *d_descriptor1;
        uint4 *d_descriptor2;

        PointCoordinates *d_world_coordinates_tmp;
        PointCoordinates *d_camera_coordinates_tmp;
        float2 *d_camera_image_coordinates_tmp;
        uint *d_num_of_measurements_tmp;
        uint *d_global_id_tmp;
        uint4 *d_descriptor1_tmp;
        uint4 *d_descriptor2_tmp;

        std::vector<Measurement *> allocated_measurements_memory;

        int size = 0;
        int allocated_size = STARTING_NUM_OF_LANDMARKS;
        Measurement measurements_template[MAX_NUM_OF_MEASUREMENTS];

        Landmarks()
        {
            allocate_memory(STARTING_NUM_OF_LANDMARKS);

            for (int i = 0; i < MAX_NUM_OF_MEASUREMENTS; i++)
            {
                measurements_template[i] = Measurement();
            }
        }

        ~Landmarks()
        {
            free_memory();
            free_tmp_memory();

            for (int i = 0; i < allocated_measurements_memory.size(); i++)
            {
                cudaFree(allocated_measurements_memory[i]);
            }
            allocated_measurements_memory.clear();
        };

        void allocate_memory(int num)
        {
            checkCudaErrors(cudaMalloc((void **)&d_world_coordinates,
                                       sizeof(PointCoordinates) * num));
            checkCudaErrors(cudaMalloc((void **)&d_camera_coordinates,
                                       sizeof(PointCoordinates) * num));
            checkCudaErrors(cudaMalloc((void **)&d_camera_image_coordinates,
                                       sizeof(float2) * num));
            checkCudaErrors(cudaMalloc((void **)&d_num_of_measurements,
                                       sizeof(uint) * num));
            checkCudaErrors(cudaMalloc((void **)&d_global_id,
                                       sizeof(uint) * num));
            checkCudaErrors(cudaMalloc((void **)&d_descriptor1,
                                       sizeof(uint4) * num));
            checkCudaErrors(cudaMalloc((void **)&d_descriptor2,
                                       sizeof(uint4) * num));
        }

        void free_memory()
        {
            if (d_world_coordinates)
                checkCudaErrors(cudaFree(d_world_coordinates));
            if (d_camera_coordinates)
                checkCudaErrors(cudaFree(d_camera_coordinates));
            if (d_camera_image_coordinates)
                checkCudaErrors(cudaFree(d_camera_image_coordinates));
            if (d_num_of_measurements)
                checkCudaErrors(cudaFree(d_num_of_measurements));
            if (d_global_id)
                checkCudaErrors(cudaFree(d_global_id));
            if (d_descriptor1)
                checkCudaErrors(cudaFree(d_descriptor1));
            if (d_descriptor2)
                checkCudaErrors(cudaFree(d_descriptor2));
        }

        void free_tmp_memory()
        {
            if (d_world_coordinates_tmp)
                checkCudaErrors(cudaFree(d_world_coordinates_tmp));
            if (d_camera_coordinates_tmp)
                checkCudaErrors(cudaFree(d_camera_coordinates_tmp));
            if (d_camera_image_coordinates_tmp)
                checkCudaErrors(cudaFree(d_camera_image_coordinates_tmp));
            if (d_num_of_measurements_tmp)
                checkCudaErrors(cudaFree(d_num_of_measurements_tmp));
            if (d_global_id_tmp)
                checkCudaErrors(cudaFree(d_global_id_tmp));
            if (d_descriptor1_tmp)
                checkCudaErrors(cudaFree(d_descriptor1_tmp));
            if (d_descriptor2_tmp)
                checkCudaErrors(cudaFree(d_descriptor2));
        }

        // recreates arrays, all data will be lost
        void resize(int new_size, cudaStream_t stream)
        {
            if (allocated_measurements_memory.size() > 0)
            {
                for (int i = 0; i < allocated_measurements_memory.size(); i++)
                {
                    cudaFree(allocated_measurements_memory[i]);
                }
                allocated_measurements_memory.clear();
            }
            free_memory();
            free_tmp_memory();

            size = 0;
            allocated_size = 0;

            expand(new_size, stream);
        }

        // increase storage by allocating new memory and copying old
        // data to new arrays
        void expand(cudaStream_t stream)
        {
            expand(allocated_size, stream);
        }

        void expand(int num_of_additional_landmarks, cudaStream_t stream)
        {
            free_tmp_memory();

            d_world_coordinates = d_world_coordinates_tmp;
            d_camera_coordinates = d_camera_coordinates_tmp;
            d_camera_image_coordinates = d_camera_image_coordinates_tmp;
            d_num_of_measurements = d_num_of_measurements_tmp;
            d_global_id = d_global_id_tmp;
            d_descriptor1 = d_descriptor1_tmp;
            d_descriptor2 = d_descriptor2_tmp;

            int newSize = allocated_size + num_of_additional_landmarks;

            allocate_memory(newSize);

            if (d_world_coordinates_tmp)
            {
                checkCudaErrors(cudaMemcpyAsync((void *)&d_world_coordinates,
                                                (void *)&d_world_coordinates_tmp,
                                                sizeof(PointCoordinates) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
                checkCudaErrors(cudaMemcpyAsync((void *)&d_camera_coordinates,
                                                (void *)&d_camera_coordinates_tmp,
                                                sizeof(PointCoordinates) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
                checkCudaErrors(cudaMemcpyAsync((void *)&d_camera_image_coordinates,
                                                (void *)&d_camera_image_coordinates_tmp,
                                                sizeof(float2) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
                checkCudaErrors(cudaMemcpyAsync((void *)&d_num_of_measurements,
                                                (void *)&d_num_of_measurements_tmp,
                                                sizeof(uint) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
                checkCudaErrors(cudaMemcpyAsync((void *)&d_global_id,
                                                (void *)&d_global_id_tmp,
                                                sizeof(uint) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
                checkCudaErrors(cudaMemcpyAsync((void *)&d_descriptor1,
                                                (void *)&d_descriptor1_tmp,
                                                sizeof(uint4) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
                checkCudaErrors(cudaMemcpyAsync((void *)&d_descriptor2,
                                                (void *)&d_descriptor2_tmp,
                                                sizeof(uint4) * newSize,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
            }

            if (allocated_measurements_memory.size() > 0)
                allocate_measurements(stream);

            allocated_size += num_of_additional_landmarks;
        }

        // this is a separate function and is not in constructor as visible landmarks
        // do not need to allocate measurements
        void allocate_measurements(cudaStream_t stream)
        {
            for (int i = allocated_measurements_memory.size(); i < allocated_size; i++)
            {
                Measurement *measurements;
                checkCudaErrors(cudaMalloc((void **)&measurements,
                                           sizeof(Measurement) * MAX_NUM_OF_MEASUREMENTS));
                checkCudaErrors(cudaMemcpyAsync((void *)measurements,
                                                (void *)measurements_template,
                                                sizeof(Measurement) * MAX_NUM_OF_MEASUREMENTS,
                                                cudaMemcpyHostToDevice,
                                                stream));

                allocated_measurements_memory.push_back(measurements);
            }
        }

    } Landmarks_t;

}

#endif // JETRACER_CUDA_LANDMARKS_H
