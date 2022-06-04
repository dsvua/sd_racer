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

        // Realsense D400 events
        event_realsense_D400_rgb,
        event_realsense_D400_rgbd,
        event_realsense_D400_accel,
        event_realsense_D400_gyro,

        // GPU events
        event_gpu_callback,
        event_gpu_slam_frame,
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