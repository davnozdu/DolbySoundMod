#pragma once

#include <fmq/AidlMessageQueue.h>
#include <fmq/EventFlag.h>

#include <aidl/android/hardware/common/fmq/MQDescriptor.h>
#include <aidl/android/hardware/common/fmq/SynchronizedReadWrite.h>
#include <android/log.h>
#include <unistd.h>
#include <utils/Errors.h>

#include <cstdint>
#include <memory>

#ifndef ALOGE
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "AllySoundAidl", __VA_ARGS__)
#endif
#ifndef ALOGW
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, "AllySoundAidl", __VA_ARGS__)
#endif
#ifndef ALOGD
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "AllySoundAidl", __VA_ARGS__)
#endif

namespace ally {

using aidl::android::hardware::common::fmq::MQDescriptor;
using aidl::android::hardware::common::fmq::SynchronizedReadWrite;

// Thin wrapper over AOSP libfmq (third_party/libfmq) instead of a hand-rolled ring
// buffer. The grantors, the queue size, the page alignment of the backing memory and
// the event flag word are then produced by the very code AudioFlinger uses to consume
// the queues, so the two sides cannot disagree about the layout. A hand-built layout
// that looked identical still left AudioFlinger with an invalid status queue, and it
// gave up silently with 'invalid efGroup' on every buffer.
template <typename T>
class FmqQueue {
public:
    using Queue = android::AidlMessageQueue<T, SynchronizedReadWrite>;
    using Descriptor = MQDescriptor<T, SynchronizedReadWrite>;

    FmqQueue() = default;
    ~FmqQueue() { reset(); }

    FmqQueue(const FmqQueue&) = delete;
    FmqQueue& operator=(const FmqQueue&) = delete;

    // with_event_flag: the status queue carries the event flag word AudioFlinger wakes
    // on; the data queues of the stock effect libraries do not have one (3 grantors).
    bool init(size_t count, bool with_event_flag = true) {
        reset();
        queue_ = std::make_unique<Queue>(count, with_event_flag);
        if (!queue_ || !queue_->isValid()) {
            ALOGE("fmq: queue init failed (count=%zu element=%zu evflag=%d)", count, sizeof(T),
                  (int)with_event_flag);
            queue_.reset();
            return false;
        }
        if (auto* word = queue_->getEventFlagWord()) {
            android::hardware::EventFlag::createEventFlag(word, &ef_);
        }
        if (with_event_flag && !ef_) {
            ALOGE("fmq: event flag init failed");
            reset();
            return false;
        }
        return true;
    }

    void reset() {
        if (ef_) android::hardware::EventFlag::deleteEventFlag(&ef_);
        ef_ = nullptr;
        queue_.reset();
    }

    bool valid() const { return queue_ && queue_->isValid(); }

    Descriptor dup_descriptor() const { return queue_ ? queue_->dupeDesc() : Descriptor(); }

    size_t available_to_read() const { return queue_ ? queue_->availableToRead() : 0; }
    size_t available_to_write() const { return queue_ ? queue_->availableToWrite() : 0; }
    size_t capacity() const { return queue_ ? queue_->getQuantumCount() : 0; }

    bool read(T* out, size_t n) { return queue_ && queue_->read(out, n); }
    bool write(const T* in, size_t n) { return queue_ && queue_->write(in, n); }

    void wake(uint32_t bitmask = 1) {
        if (ef_) ef_->wake(bitmask);
    }

    bool wait(uint32_t bitmask, int timeout_ms) {
        if (!ef_) return false;
        uint32_t ef_state = 0;
        return ef_->wait(bitmask, &ef_state, static_cast<int64_t>(timeout_ms) * 1000000LL,
                         true /* retry */) == ::android::OK;
    }

private:
    std::unique_ptr<Queue> queue_;
    android::hardware::EventFlag* ef_ = nullptr;
};

}  // namespace ally
