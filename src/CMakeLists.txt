ADD_LIBRARY(${PROJECT_NAME}_library_gpu
    Cuda/CudaCommon.cpp
    Types/Fault.cpp
    Types/ORUtils/SE3Pose.cpp
    Cameras/RealSenseD400/RealSenseD400.cpp
    Cameras/RealSenseD400/Cuda/DepthAlign.cu
    EventLoop/EventsThread.cpp
    EventLoop/MainEventLoop.cpp
    WebSocket/FoxgloveWebSocketCom.cpp
    Slam/SlamLoop.cpp
    Slam/TrackPoints.cpp
    Slam/ORB/OrbDetector.cpp
    Slam/ORB/Cuda/Fast.cu
    Slam/ORB/Cuda/Orb.cu
    Slam/ORB/Cuda/Nms.cu
    Slam/ORB/Cuda/RgbToGrayscale.cu
)

ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "../include")
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_library_gpu ${realsense2_LIBRARY} nlohmann_json::nlohmann_json -pthread)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_library_gpu PUBLIC ${realsense2_LIBRARY} nppif nvjpeg -pthread)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME}_library_gpu PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "../include")
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
SET_TARGET_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_SOURCE_FILES_PROPERTIES(${PROJECT_NAME}_library_gpu PROPERTIES LANGUAGE CUDA)

INCLUDE_DIRECTORIES("/root/cuda-samples/Common" Eigen3::Eigen)
