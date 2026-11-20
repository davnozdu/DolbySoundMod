#pragma once

#include <aidl/android/hardware/common/fmq/MQDescriptor.h>
#include <aidl/android/hardware/common/fmq/SynchronizedReadWrite.h>
#include <android/binder_ibinder.h>
#include <android/log.h>

#ifndef ALOGE
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "AllySoundAidl", __VA_ARGS__)
#endif
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <vector>

namespace ally {

using aidl::android::hardware::common::NativeHandle;
using aidl::android::hardware::common::fmq::GrantorDescriptor;
using aidl::android::hardware::common::fmq::MQDescriptor;
using aidl::android::hardware::common::fmq::SynchronizedReadWrite;

// Minimal SynchronizedReadWrite FMQ compatible with AOSP AidlMessageQueue.
template <typename T>
class FmqQueue {
public:
    FmqQueue() = default;
    ~FmqQueue() { reset(); }

    FmqQueue(const FmqQueue&) = delete;
    FmqQueue& operator=(const FmqQueue&) = delete;

    bool init(size_t count) {
        reset();
        count_ = count;
        data_bytes_ = count * sizeof(T);
        size_t aligned_data = (data_bytes_ + 7) & ~size_t(7);
        total_bytes_ = 16 + aligned_data + 4; // readPtr(8) + writePtr(8) + data + eventFlag(4)

        errno = 0;
        fd_ = static_cast<int>(syscall(__NR_memfd_create, "ally_fmq", 1 /* MFD_CLOEXEC */));
        if (fd_ < 0) {
            ALOGE("fmq: memfd_create failed: %s (count=%zu bytes=%zu)", strerror(errno), count_, total_bytes_);
            return false;
        }
        if (ftruncate(fd_, static_cast<off_t>(total_bytes_)) != 0) {
            ALOGE("fmq: ftruncate failed: %s (bytes=%zu)", strerror(errno), total_bytes_);
            reset();
            return false;
        }
        base_ = static_cast<uint8_t*>(
            mmap(nullptr, total_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
        if (base_ == MAP_FAILED) {
            ALOGE("fmq: mmap failed: %s (bytes=%zu)", strerror(errno), total_bytes_);
            base_ = nullptr;
            reset();
            return false;
        }
        std::memset(base_, 0, total_bytes_);
        read_ptr_ = reinterpret_cast<std::atomic<uint64_t>*>(base_ + 0);
        write_ptr_ = reinterpret_cast<std::atomic<uint64_t>*>(base_ + 8);
        data_ = base_ + 16;
        event_flag_ = reinterpret_cast<std::atomic<uint32_t>*>(base_ + 16 + aligned_data);
        return true;
    }

    void reset() {
        if (base_) {
            munmap(base_, total_bytes_);
            base_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
        read_ptr_ = nullptr;
        write_ptr_ = nullptr;
        data_ = nullptr;
        event_flag_ = nullptr;
        count_ = 0;
        data_bytes_ = 0;
        total_bytes_ = 0;
    }

    MQDescriptor<T, SynchronizedReadWrite> dup_descriptor() const {
        MQDescriptor<T, SynchronizedReadWrite> desc;
        desc.quantum = static_cast<int32_t>(sizeof(T));
        desc.flags = 1; // SynchronizedReadWrite
        desc.grantors.resize(4);
        desc.grantors[0] = {0, 0, 8};
        desc.grantors[1] = {0, 8, 8};
        desc.grantors[2] = {0, 16, static_cast<int64_t>(data_bytes_)};
        size_t aligned_data = (data_bytes_ + 7) & ~size_t(7);
        desc.grantors[3] = {0, static_cast<int32_t>(16 + aligned_data), 4};

        int dup_fd = fcntl(fd_, F_DUPFD_CLOEXEC, 0);
        desc.handle.fds.emplace_back(ndk::ScopedFileDescriptor(dup_fd));
        return desc;
    }

    size_t available_to_read() const {
        if (!read_ptr_ || !write_ptr_) return 0;
        uint64_t w = write_ptr_->load(std::memory_order_acquire);
        uint64_t r = read_ptr_->load(std::memory_order_relaxed);
        return static_cast<size_t>((w - r) / sizeof(T));
    }

    size_t available_to_write() const {
        if (!read_ptr_ || !write_ptr_) return 0;
        uint64_t r = read_ptr_->load(std::memory_order_acquire);
        uint64_t w = write_ptr_->load(std::memory_order_relaxed);
        return count_ - static_cast<size_t>((w - r) / sizeof(T));
    }

    bool read(T* out, size_t n) {
        if (available_to_read() < n) return false;
        uint64_t r = read_ptr_->load(std::memory_order_relaxed);
        size_t byte_off = static_cast<size_t>(r % data_bytes_);
        size_t bytes = n * sizeof(T);
        size_t first = std::min(bytes, data_bytes_ - byte_off);
        std::memcpy(out, data_ + byte_off, first);
        if (first < bytes) {
            std::memcpy(reinterpret_cast<uint8_t*>(out) + first, data_, bytes - first);
        }
        read_ptr_->store(r + bytes, std::memory_order_release);
        return true;
    }

    bool write(const T* in, size_t n) {
        if (available_to_write() < n) return false;
        uint64_t w = write_ptr_->load(std::memory_order_relaxed);
        size_t byte_off = static_cast<size_t>(w % data_bytes_);
        size_t bytes = n * sizeof(T);
        size_t first = std::min(bytes, data_bytes_ - byte_off);
        std::memcpy(data_ + byte_off, in, first);
        if (first < bytes) {
            std::memcpy(data_, reinterpret_cast<const uint8_t*>(in) + first, bytes - first);
        }
        write_ptr_->store(w + bytes, std::memory_order_release);
        return true;
    }

    void wake(uint32_t bitmask = 1) {
        if (!event_flag_) return;
        event_flag_->fetch_or(bitmask, std::memory_order_release);
        syscall(__NR_futex, reinterpret_cast<uint32_t*>(event_flag_), FUTEX_WAKE, INT32_MAX, nullptr, nullptr, 0);
    }

    bool wait(uint32_t bitmask = 1, int timeout_ms = 50) {
        if (!event_flag_) return false;
        uint32_t cur = event_flag_->load(std::memory_order_acquire);
        if (cur & bitmask) {
            event_flag_->fetch_and(~bitmask, std::memory_order_acq_rel);
            return true;
        }
        struct timespec ts {};
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000L;
        syscall(__NR_futex, reinterpret_cast<uint32_t*>(event_flag_), FUTEX_WAIT, cur, &ts, nullptr, 0);
        cur = event_flag_->load(std::memory_order_acquire);
        if (cur & bitmask) {
            event_flag_->fetch_and(~bitmask, std::memory_order_acq_rel);
            return true;
        }
        return false;
    }

private:
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    size_t count_ = 0;
    size_t data_bytes_ = 0;
    size_t total_bytes_ = 0;
    std::atomic<uint64_t>* read_ptr_ = nullptr;
    std::atomic<uint64_t>* write_ptr_ = nullptr;
    uint8_t* data_ = nullptr;
    std::atomic<uint32_t>* event_flag_ = nullptr;
};

} // namespace ally
