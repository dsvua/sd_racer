#ifndef JETRACER_MAIN_EventLoop_THREAD_H
#define JETRACER_MAIN_EventLoop_THREAD_H

#include <iostream>
#include <mutex>
#include <map>

#include "../EventLoop/EventsThread.h"
#include "../Types/Constants.h"
#include "../Types/Context.h"
#include "../Types/Events.h"

namespace Jetracer
{

    class MainEventLoop : public EventsThread
    {
    public:
        MainEventLoop(const std::string threadName, context_t *ctx);
        // ~MainEventsLoop();

        bool subscribeForEvent(EventType _event_type, std::string _thread_name,
                               std::function<bool(pEvent)> pushEventCallback);
        bool unSubscribeFromEvent(EventType _event_type, std::string _thread_name);

    private:
        void handleEvent(pEvent event);

        context_t *_ctx;
        std::mutex m_mutex_subscribers;
        std::map<EventType, std::map<std::string, std::function<bool(pEvent)>>> _subscribers;
        std::vector<Jetracer::EventsThread *> _started_threads;
    };
} // namespace Jetracer

#endif // JETRACER_MAIN_EventLoop_THREAD_H
