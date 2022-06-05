#include "Slam/SlamLoop.h"
#include "Slam/ORB/Cuda/RgbToGrayscale.h"
#include "Slam/ORB/OrbDetector.h"

#include <memory>
#include <chrono>
#include <iostream>

// using namespace std;

namespace Jetracer
{
    SlamLoop::SlamLoop(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {
        auto pushEventCallback = [this](pEvent event) -> bool
        {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_rgbd_frame, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_stereo_frame, threadName, pushEventCallback);

        std::cout << "SlamLoop is initialized" << std::endl;
    }

    void SlamLoop::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_rgbd_frame:
        {
            processRgbdFrame(std::static_pointer_cast<RgbdFrame>(event->message));
            break;
        }

        case EventType::event_stop_thread:
        {
            break;
        }

        default:
        {
            break;
        }
        }
    }

    void SlamLoop::processRgbdFrame(pRgbdFrame rgbd_frame)
    {
        auto base_frame = std::static_pointer_cast<BaseFrame>(rgbd_frame);
        rgbd_frame->uploadToGPU(tmp_frame.stream);

        if (!rgbd_frame->depth_aligned)
        {
            // std::cout << "Aligning Depth" << std::endl;
            rgbd_frame->depthFrameAligner(base_frame, tmp_frame);
            std::cout << "Frame ID: " << rgbd_frame->rgb_frame_id << std::endl;
        }
        else
        {
            std::cout << "Frame is already aligned" << std::endl;
        }

        if (!rgbd_frame->keypoints_detected)
        {
            rgb_to_grayscale(rgbd_frame, tmp_frame);
            detectOrbs(rgbd_frame, tmp_frame);

            checkCudaErrors(cudaStreamSynchronize(tmp_frame.stream));
            tmp_frame.previous_frame = base_frame;
        }
    }

} // namespace Jetracer