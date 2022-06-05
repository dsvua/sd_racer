#ifndef JETRACER_REALSENSE_D400_THREAD_H
#define JETRACER_REALSENSE_D400_THREAD_H

#include <iostream>

#include "../../EventLoop/EventsThread.h"
#include "../../Types/Context.h"
#include "../../Types/Events.h"
#include "../../Types/Frames.h"
#include "Cuda/DepthAlign.h"
#include <mutex>
#include <atomic>
#include <thread>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <cuda_runtime.h>
#include <chrono>

namespace Jetracer
{

    class RealSenseD400 : public EventsThread
    {
    public:
        RealSenseD400(const std::string threadName, context_t *ctx);
        void depth_align(RgbdFrame_t rgbd_frame);
        // ~RealSenseD400();

    private:
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;

        rs2_intrinsics intrinsics;
        rs2::config cfg;
        rs2::pipeline pipe;
        rs2::pipeline_profile selection;
    };

    typedef struct RealSenseD400RgbdFrame : RgbdFrame
    {
        rs2_intrinsics rgb_intrinsics;
        rs2_intrinsics depth_intrinsics;
        rs2_extrinsics extrinsics;
        float depth_scale;

    } RealSenseD400RgbdFrame_t;

    typedef std::shared_ptr<Jetracer::RealSenseD400RgbdFrame> pRealSenseD400RgbdFrame;

} // namespace Jetracer

#endif // JETRACER_REALSENSE_D400_THREAD_H
