#ifndef JETRACER_FOXGLOVE_WEBSOCKET_H
#define JETRACER_FOXGLOVE_WEBSOCKET_H

#include "WebSocket/foxglove/websocket/server.h"
#include <nlohmann/json.hpp>
#include <thread>

#include "EventLoop/EventsThread.h"
#include "Types/Context.h"
#include "Types/Events.h"
#include "Types/Frames.h"

using json = nlohmann::json;

namespace Jetracer
{
    class FoxgloveWebSocketCom : public EventsThread
    {
    public:
        FoxgloveWebSocketCom(const std::string threadName, context_t *ctx);
        // ~FoxgloveWebSocketCom();
    private:
        void handleEvent(pEvent event);
        void Communication();
        uint64_t nanosecondsSinceEpoch();
        void normalizeSecNSec(uint64_t &sec, uint64_t &nsec);
        std::string encodeImageBase64(size_t in_len, unsigned char *image);

        context_t *_ctx;
        foxglove::websocket::Server server;
        uint32_t imageChanId;
        std::thread *CommunicationThread;
    };
} // namespace Jetracer

#endif // JETRACER_FOXGLOVE_WEBSOCKET_H