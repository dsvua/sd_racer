#include "Cameras/RealSenseD400/RealSenseD400.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <algorithm> // for std::find_if
#include <cstring>
#include <librealsense2/rs_advanced_mode.hpp>

// using namespace std;

namespace Jetracer
{
    float get_depth_scale(rs2::device dev);
    rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile> &streams);
    bool profile_changed(const std::vector<rs2::stream_profile> &current, const std::vector<rs2::stream_profile> &prev);

    RealSenseD400::RealSenseD400(const std::string threadName, context_t *ctx) : EventsThread(threadName), _ctx(ctx)
    {

        // callback for new frames as here: https://github.com/IntelRealSense/librealsense/blob/master/examples/callback/rs-callback.cpp
        auto callbackNewFrame = [this](const rs2::frame &frame)
        {
            rs2::frameset fs = frame.as<rs2::frameset>();
            pEvent event = std::make_shared<BaseEvent>();

            if (fs)
            {
                // With callbacks, all synchronized streams will arrive in a single frameset
                // for (auto &&fr : fs)
                // {
                //     std::cout << " " << fr.get_profile().stream_name(); // will print: Depth Infrared 1
                // }
                // std::cout << std::endl;

                auto rs_depth_frame = fs.get_depth_frame();
                auto rs_rgb_frame = fs.get_color_frame();
                auto image_height = rs_rgb_frame.get_height();
                auto image_width = rs_rgb_frame.get_width();

                auto rgbd_frame = std::make_shared<RealSenseD400RgbdFrame_t>();
                rgbd_frame->frame_type = FrameType::RGBD;
                rgbd_frame->depthFrameAligner = &depthAligner;

                rgbd_frame->rgb_image_resolution.x = image_width;
                rgbd_frame->rgb_image_resolution.y = image_height;

                rgbd_frame->depth_image_resolution.x = rs_depth_frame.get_width();
                rgbd_frame->depth_image_resolution.y = rs_depth_frame.get_height();

                rgbd_frame->timestamp = rs_rgb_frame.get_frame_metadata(RS2_FRAME_METADATA_BACKEND_TIMESTAMP);

                rgbd_frame->depth_frame_id = rs_depth_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
                rgbd_frame->rgb_frame_id = rs_rgb_frame.get_frame_number();

                rgbd_frame->depth_image_size = rs_depth_frame.get_data_size();
                rgbd_frame->rgb_image_size = rs_rgb_frame.get_data_size();

                auto depth_profile = rs_depth_frame.get_profile().as<rs2::video_stream_profile>();
                auto rgb_profile = rs_rgb_frame.get_profile().as<rs2::video_stream_profile>();

                rgbd_frame->h_depth_image = (uint16_t *)malloc(rgbd_frame->depth_image_size * sizeof(char));
                rgbd_frame->h_rgb_image = (unsigned char *)malloc(rgbd_frame->rgb_image_size * sizeof(char));

                rgbd_frame->depth_intrinsics = depth_profile.get_intrinsics();
                rgbd_frame->rgb_intrinsics = rgb_profile.get_intrinsics();
                rgbd_frame->extrinsics = depth_profile.get_extrinsics_to(rgb_profile);
                rgbd_frame->depth_scale = rs_depth_frame.get_units();

                std::memcpy(rgbd_frame->h_depth_image, rs_depth_frame.get_data(), rgbd_frame->depth_image_size);
                std::memcpy(rgbd_frame->h_rgb_image, rs_rgb_frame.get_data(), rgbd_frame->rgb_image_size);

                rgbd_frame->camera_matrix << rgbd_frame->rgb_intrinsics.fx, 0, rgbd_frame->rgb_intrinsics.ppx,
                    0, rgbd_frame->rgb_intrinsics.fy, rgbd_frame->rgb_intrinsics.ppy,
                    0, 0, 0;

                event->event_type = EventType::event_rgbd_frame;
                event->message = rgbd_frame;
                this->_ctx->sendEvent(event);
                // std::cout << "Frame sent: " << rgbd_frame->rgb_frame_id << std::endl;
            }
            else
            {
                // std::cout << " " << frame.get_profile().stream_name() << std::endl;
                switch (frame.get_profile().stream_type())
                {
                case RS2_STREAM_GYRO:
                {
                    auto motion = frame.as<rs2::motion_frame>();

                    ImuFrame_t imu_frame;
                    imu_frame.motion_data = motion.get_motion_data();
                    imu_frame.timestamp = motion.get_timestamp();
                    imu_frame.frame_type = FrameType::GYRO;

                    event->event_type = EventType::event_imu_gyro;
                    event->message = std::make_shared<ImuFrame_t>(imu_frame);

                    this->_ctx->sendEvent(event);
                    break;
                }

                case RS2_STREAM_ACCEL:
                {
                    auto motion = frame.as<rs2::motion_frame>();

                    ImuFrame_t imu_frame;
                    imu_frame.motion_data = motion.get_motion_data();
                    imu_frame.timestamp = motion.get_timestamp();
                    imu_frame.frame_type = FrameType::ACCEL;

                    event->event_type = EventType::event_imu_accel;
                    event->message = std::make_shared<ImuFrame_t>(imu_frame);

                    this->_ctx->sendEvent(event);
                    break;
                }

                default:
                    break;
                }
            }

            // std::cout << std::endl;
        };

        auto pushEventCallback = [this](pEvent event) -> bool
        {
            this->pushEvent(event);
            return true;
        };

        _ctx->subscribeForEvent(EventType::event_stop_thread, threadName, pushEventCallback);

        // Add desired streams to configuration
        cfg.enable_stream(RS2_STREAM_COLOR, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_RGB8, _ctx->fps);
        cfg.enable_stream(RS2_STREAM_DEPTH, _ctx->cam_w, _ctx->cam_h, RS2_FORMAT_Z16, _ctx->fps);
        cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 63); // 63 or 250
        cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200); // 200 or 400

        // Start the camera pipeline
        selection = pipe.start(cfg, callbackNewFrame);

        // disabling laser
        rs2::device selected_device = selection.get_device();
        auto depth_sensor = selected_device.first<rs2::depth_sensor>();
        // depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter/laser

        // Auto-exposure priority should be disabled to get 60fps on color camera
        auto color_sensor = selected_device.first<rs2::color_sensor>();
        color_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
        color_sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_PRIORITY, 0);

        std::cout << "RealSenseD400 is initialized" << std::endl;
    }

    void RealSenseD400::handleEvent(pEvent event)
    {

        switch (event->event_type)
        {

        case EventType::event_stop_thread:
        {
            std::cout << "Stopping Realsense pipeline " << std::endl;
            pipe.stop();
            std::cout << "Stopped Realsense pipeline " << std::endl;
            break;
        }

        default:
        {
            break;
        }
        }
    }

} // namespace Jetracer