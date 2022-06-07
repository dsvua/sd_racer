#ifndef JETRACER_TEMPLATE_THREAD_H
#define JETRACER_TEMPLATE_THREAD_H

#include <iostream>

#include "../EventLoop/EventsThread.h"
#include "../Types/Context.h"
#include "../Types/Events.h"
#include "../Types/Frames.h"
#include "../Types/Landmarks.h"
#include <mutex>
#include <atomic>
#include <thread>

namespace Jetracer
{

    class SlamLoop : public EventsThread
    {
    public:
        SlamLoop(const std::string threadName, context_t *ctx);

        // ~SlamLoop();
        ;

    private:
        void handleEvent(pEvent event);
        void processRgbdFrame(pRgbdFrame rgbd_frame);
        void trackPoints(pRgbdFrame current_frame);
        // void updatePoints(_context, current_frame);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        TmpData_t tmp_frame;
        Matrix4f robot_to_world;
        Landmarks landmarks;
        // TransformMatrix3D robot_to_world = TransformMatrix3D::Identity();
    };

} // namespace Jetracer

#endif // JETRACER_TEMPLATE_THREAD_H
