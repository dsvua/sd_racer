#ifndef JETRACER_EVENTS_H
#define JETRACER_EVENTS_H

#include <memory>

namespace Jetracer
{
    // when adding new EventType do not forget to add it
    enum class EventType
    {
        // thread events
        event_start_thread,
        event_stop_thread,

        // keep alive
        event_ping,
        event_pong,

        // Camera events
        event_rgbd_frame,
        event_stereo_frame,

        // IMU events, could be emitted by camera
        event_imu_gyro,
        event_imu_accel,

        // GPU events
        event_gpu_callback,

        // SLAM events
        event_rgbd_slam_processed,
        event_stereo_slam_processed,
    };

    typedef std::shared_ptr<void> pMessage;

    class BaseEvent
    {
    public:
        EventType event_type;
        pMessage message;
    };

    typedef std::shared_ptr<BaseEvent> pEvent;

    class ThreadEvent : public BaseEvent
    {
    public:
        std::string thread_name;
    };

    typedef std::shared_ptr<ThreadEvent> pThreadEvent; // p - means pointer

} // namespace Jetracer

#endif // JETRACER_EVENTS_H