
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)


message(STATUS "CUDA_cudadevrt_LIBRARY ${CUDA_cudadevrt_LIBRARY}")
message(STATUS "CUDADEVRT_LIBRARY ${CUDADEVRT_LIBRARY}")

INCLUDE_DIRECTORIES("/root/cuda-samples/Common" Eigen3::Eigen)
