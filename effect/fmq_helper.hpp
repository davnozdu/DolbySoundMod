#pragma once

#include <aidl/android/hardware/common/fmq/MQDescriptor.h>
#include <aidl/android/hardware/common/fmq/SynchronizedReadWrite.h>
#include <android/binder_ibinder.h>
#include <sys/ioctl.h>

// Weak on purpose: linking libandroid.so directly makes the whole effect library
// unloadable in a vendor process (its namespace does not expose system-only libs).
extern "C" int ASharedMemory_create(const char* name, size_t size) __attribute__((weak));
#include <android/log.h>

#ifndef ALOGE
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "AllySoundAidl", __VA_ARGS__)
#endif
#ifndef ALOGW
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, "AllySoundAidl", __VA_ARGS__)
#endif
#ifndef ALOGD
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "AllySoundAidl", __VA_ARGS__)
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

    // Legacy but universally allowed path for vendor processes: /dev/ashmem.
    static int open_ashmem(const char* name, size_t bytes) {
        int fd = open("/dev/ashmem", O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            ALOGW("fmq: open /dev/ashmem: %s", strerror(errno));
            return -1;
        }
        if (ioctl(fd, _IOW('a', 0x21, char*), name) != 0) {
            ALOGW("fmq: ashmem set name: %s", strerror(errno));
        }
        if (ioctl(fd, _IOW('a', 0x20, size_t), bytes) != 0) {
            ALOGW("fmq: ashmem set size: %s", strerror(errno));
            close(fd);
            return -1;
        }
        return fd;
    }

    // A vendor HAL domain may refuse ftruncate on a memfd while still allowing other
    // ways to give it a size, so try the alternatives before giving up.
    static bool size_file(int fd, size_t bytes) {
        if (ftruncate(fd, static_cast<off_t>(bytes)) == 0) return true;
        ALOGW("fmq: ftruncate errno=%d, trying fallocate", errno);
        if (fallocate(fd, 0, 0, static_cast<off_t>(bytes)) == 0) return true;
        ALOGW("fmq: fallocate errno=%d, trying pwrite", errno);
        static const size_t kChunk = 65536;
        std::vector<uint8_t> zeros(std::min(bytes, kChunk), 0);
        size_t done = 0;
        while (done < bytes) {
            size_t n = std::min(zeros.size(), bytes - done);
            ssize_t w = pwrite(fd, zeros.data(), n, static_cast<off_t>(done));
            if (w <= 0) {
                ALOGE("fmq: pwrite errno=%d at %zu/%zu", errno, done, bytes);
                return false;
            }
            done += static_cast<size_t>(w);
        }
        ALOGW("fmq: sized memfd with pwrite (%zu bytes)", bytes);
        return true;
    }

    bool init(size_t count) {
        reset();
        count_ = count;
        data_bytes_ = count * sizeof(T);
        size_t aligned_data = (data_bytes_ + 7) & ~size_t(7);
        total_bytes_ = 16 + aligned_data + 4; // readPtr(8) + writePtr(8) + data + eventFlag(4)

        errno = 0;
        // Prefer the platform allocator: it is the same path libfmq uses for the stock
        // effect libraries in this very process, so whatever the HAL domain allows for
        // them it allows here (a bare memfd + ftruncate hits EACCES in that domain).
        bool sized = false;
        if (ASharedMemory_create != nullptr) {
            fd_ = ASharedMemory_create("ally_fmq", total_bytes_);
            if (fd_ >= 0) sized = true;
        }
        if (!sized) {
            fd_ = open_ashmem("ally_fmq", total_bytes_);
            if (fd_ >= 0) {
                sized = true;
                ALOGD("fmq: queue from /dev/ashmem (%zu bytes)", total_bytes_);
            }
        }
        if (!sized) {
            ALOGW("fmq: ASharedMemory_create failed: %s, falling back to memfd",
                  strerror(errno));
            fd_ = static_cast<int>(syscall(__NR_memfd_create, "ally_fmq", 1 /* MFD_CLOEXEC */));
            if (fd_ < 0) {
                ALOGE("fmq: memfd_create failed: %s (count=%zu bytes=%zu)", strerror(errno),
                      count_, total_bytes_);
                return false;
            }
        }
        if (!sized && !size_file(fd_, total_bytes_)) {
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
        // libfmq's MQDescriptor<T, flavor>::isValid() compares this against the
        // flavor template parameter, where kSynchronizedReadWrite == 0. Sending 1
        // makes the client build an *unsynchronized* queue, which has no event
        // flag word - AudioFlinger then reports "<name> invalid efGroup" on every
        // buffer and never touches our queues.
        desc.flags = 0; // SynchronizedReadWrite
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
