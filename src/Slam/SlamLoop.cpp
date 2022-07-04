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

        loadPattern(); // needed for ORB detection

        _ctx->subscribeForEvent(EventType::event_rgbd_frame, threadName, pushEventCallback);
        _ctx->subscribeForEvent(EventType::event_stereo_frame, threadName, pushEventCallback);

        checkCudaErrors(cudaStreamCreateWithFlags(&tmp_frame.stream, cudaStreamNonBlocking));
        // matrices initialization
        tmp_frame.robot_to_world.setIdentity();
        tmp_frame.worldmap_landmarks.allocate_measurements(tmp_frame.stream);

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
        if (rgbd_frame->depth_frame_id > _ctx->frames_to_skip) // camera needs to settle down for first $frames_to_skip
        {
            auto base_frame = std::static_pointer_cast<BaseFrame>(rgbd_frame);
            rgbd_frame->uploadToGPU(tmp_frame.stream);

            if (!rgbd_frame->depth_aligned)
            {
                // std::cout << "Aligning Depth" << std::endl;
                rgbd_frame->depthFrameAligner(base_frame, tmp_frame);
                std::cout << "RGB frame ID: " << rgbd_frame->rgb_frame_id << "\tDepth frame ID: " << rgbd_frame->depth_frame_id << std::endl;
            }
            else
            {
                std::cout << "Frame is already aligned" << std::endl;
            }

            if (!rgbd_frame->keypoints_detected)
            {
                rgb_to_grayscale(rgbd_frame, tmp_frame);
                detectOrbs(rgbd_frame, tmp_frame, _ctx->min_score);
                trackPoints(rgbd_frame);

                checkCudaErrors(cudaStreamSynchronize(tmp_frame.stream));
                tmp_frame.previous_frame = base_frame;
                pEvent event = std::make_shared<BaseEvent>();
                event->event_type = EventType::event_rgbd_slam_processed;
                event->message = rgbd_frame;
                this->_ctx->sendEvent(event);
            }
        }
    }

} // namespace Jetracer