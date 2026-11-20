#pragma once
#include <stdint.h>
#include <time.h>
namespace android {
enum {
    SYSTEM_TIME_REALTIME = 0,
    SYSTEM_TIME_MONOTONIC = 1,
    SYSTEM_TIME_THREAD = 2,
};
inline int64_t elapsedRealtimeNano() {
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
inline int64_t elapsedRealtime() { return elapsedRealtimeNano() / 1000000LL; }
inline int64_t uptimeMillis() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}
}  // namespace android
