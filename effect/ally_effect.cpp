#include "fmq_helper.hpp"
#include "../dsp/spatial.hpp"

#include <aidl/android/hardware/audio/effect/BnEffect.h>
#include <aidl/android/hardware/audio/effect/Descriptor.h>
#include <aidl/android/hardware/audio/effect/IEffect.h>
#include <aidl/android/media/audio/common/AudioChannelLayout.h>
#include <aidl/android/media/audio/common/AudioUuid.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define LOG_TAG "AllySoundAidl"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace aidl::android::hardware::audio::effect {

namespace {

using aidl::android::media::audio::common::AudioChannelLayout;
using aidl::android::media::audio::common::AudioUuid;

// AudioUuid holds a std::vector (not a literal type), so these are `const`,
// not `constexpr`. Every component is int32_t in the AIDL struct: casting the
// clockSeq half-word to int16_t turns 0x8000 into -32768 while the framework
// (and the HAL, which parses the config string) uses +32768, so the descriptor
// never matched the configured UUID and the HAL dropped the effect.
// Implementation UUID: a117500d-0001-4000-8000-000000000001
const AudioUuid kImplUuid = {
    static_cast<int32_t>(0xa117500d),
    0x0001,
    0x4000,
    0x8000,
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
};

// Type UUID: LoudnessEnhancer (fe3199be-aed0-413f-87bb-11260eb63cf1)
// Universally accepted by AudioFlinger on any MIXER session.
const AudioUuid kTypeUuid = {
    static_cast<int32_t>(0xfe3199be),
    0xaed0,
    0x413f,
    0x87bb,
    {0x11, 0x26, 0x0e, 0xb6, 0x3c, 0xf1},
};

int channel_count_from_layout(const AudioChannelLayout& layout) {
    if (layout.getTag() == AudioChannelLayout::layoutMask) {
        int32_t mask = layout.get<AudioChannelLayout::layoutMask>();
        return __builtin_popcount(static_cast<uint32_t>(mask));
    }
    if (layout.getTag() == AudioChannelLayout::indexMask) {
        int32_t mask = layout.get<AudioChannelLayout::indexMask>();
        return __builtin_popcount(static_cast<uint32_t>(mask));
    }
    return 2;
}

// Oplus "audiox" effect uuid (41f6c0f4-5d8f-11ec-bf63-0242ac130002): the vendor
// HAL only creates effects it can load through its own library list, so the
// module also ships this library under the Oplus name and answers for this uuid.
const AudioUuid kOplusUuid = {
    static_cast<int32_t>(0x41f6c0f4),
    0x5d8f,
    0x11ec,
    0xbf63,
    {0x02, 0x42, 0xac, 0x13, 0x00, 0x02},
};

static bool is_known_uuid(const AudioUuid& u) {
    return u == kImplUuid || u == kOplusUuid;
}

Descriptor make_descriptor(const AudioUuid& impl) {
    Descriptor desc;
    // The Oplus slot reports itself exactly like the vendor library does: its own uuid is
    // also its type uuid. AudioFlinger looks effects up by the (type, uuid) pair, so a
    // different type here makes createEffect fail with -2 even though the uuid matches.
    if (impl == kOplusUuid) {
        desc.common.id.type = kOplusUuid;
        desc.common.name = "OplusAudioX";
        desc.common.implementor = "Oplus Holdings Corp.";
    } else {
        desc.common.id.type = kTypeUuid;
        desc.common.name = "Ally Sound Lab";
        desc.common.implementor = "AllySoundKSU";
    }
    desc.common.id.uuid = impl;
    desc.common.flags.type = Flags::Type::INSERT;
    desc.common.flags.insert = Flags::Insert::FIRST;
    desc.common.flags.volume = Flags::Volume::NONE;
    desc.common.flags.hwAcceleratorMode = Flags::HardwareAccelerator::NONE;
    desc.common.flags.bypass = false;
    desc.common.flags.deviceIndication = true;
    return desc;
}

class AllySoundEffect final : public BnEffect {
public:
    explicit AllySoundEffect(const AudioUuid& impl) : impl_uuid_(impl) {}
    ~AllySoundEffect() override {
        stop_worker();
    }

    ndk::ScopedAStatus open(const Parameter::Common& common,
                            const std::optional<Parameter::Specific>& specific,
                            OpenEffectReturn* _aidl_return) override {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_worker_locked();

        common_ = common;
        sample_rate_ = static_cast<float>(common.input.base.sampleRate > 0 ? common.input.base.sampleRate : 48000);
        in_channels_ = std::max(1, channel_count_from_layout(common.input.base.channelMask));
        out_channels_ = std::max(1, channel_count_from_layout(common.output.base.channelMask));
        frame_count_ = static_cast<size_t>(std::max<int64_t>(common.input.frameCount, 256));

        renderer_.init(sample_rate_);
        renderer_.set_profile(profile_.load());
        renderer_.set_device(device_.load());
        renderer_.set_neural(neural_.load());
        renderer_.set_reverb(reverb_.load());

        if (specific.has_value()) {
            apply_specific_locked(*specific);
        }

        const size_t in_samples = std::max<size_t>(frame_count_ * in_channels_ * 4, 4096);
        const size_t out_samples = std::max<size_t>(frame_count_ * out_channels_ * 4, 4096);
        if (!status_mq_.init(1, true)) {
            ALOGE("Failed to initialize FMQ: status queue");
            return ndk::ScopedAStatus::fromExceptionCode(EX_ILLEGAL_STATE);
        }
        if (!in_mq_.init(in_samples, false)) {
            ALOGE("Failed to initialize FMQ: input queue (%zu)", in_samples);
            return ndk::ScopedAStatus::fromExceptionCode(EX_ILLEGAL_STATE);
        }
        if (!out_mq_.init(out_samples, false)) {
            ALOGE("Failed to initialize FMQ: output queue (%zu)", out_samples);
            return ndk::ScopedAStatus::fromExceptionCode(EX_ILLEGAL_STATE);
        }

        _aidl_return->statusMQ = status_mq_.dup_descriptor();
        _aidl_return->inputDataMQ = in_mq_.dup_descriptor();
        _aidl_return->outputDataMQ = out_mq_.dup_descriptor();

        state_.store(State::IDLE);
        start_worker_locked();
        ALOGI("Opened effect: sr=%.0f in_ch=%d out_ch=%d frames=%zu",
              sample_rate_, in_channels_, out_channels_, frame_count_);
        return ndk::ScopedAStatus::ok();
    }

    ndk::ScopedAStatus close() override {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_worker_locked();
        status_mq_.reset();
        in_mq_.reset();
        out_mq_.reset();
        state_.store(State::INIT);
        return ndk::ScopedAStatus::ok();
    }

    ndk::ScopedAStatus getDescriptor(Descriptor* _aidl_return) override {
        *_aidl_return = make_descriptor(impl_uuid_);
        return ndk::ScopedAStatus::ok();
    }

    ndk::ScopedAStatus command(CommandId commandId) override {
        std::lock_guard<std::mutex> lock(mutex_);
        switch (commandId) {
            case CommandId::START:
                if (state_.load() == State::IDLE) {
                    state_.store(State::PROCESSING);
                }
                break;
            case CommandId::STOP:
                if (state_.load() == State::PROCESSING) {
                    state_.store(State::IDLE);
                }
                break;
            case CommandId::RESET:
                renderer_.reset();
                break;
            default:
                break;
        }
        return ndk::ScopedAStatus::ok();
    }

    ndk::ScopedAStatus getState(State* _aidl_return) override {
        *_aidl_return = state_.load();
        return ndk::ScopedAStatus::ok();
    }

    ndk::ScopedAStatus setParameter(const Parameter& param) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (param.getTag() == Parameter::common) {
            common_ = param.get<Parameter::common>();
            return ndk::ScopedAStatus::ok();
        }
        if (param.getTag() == Parameter::specific) {
            apply_specific_locked(param.get<Parameter::specific>());
            return ndk::ScopedAStatus::ok();
        }
        return ndk::ScopedAStatus::ok();
    }

    ndk::ScopedAStatus getParameter(const Parameter::Id& paramId, Parameter* _aidl_return) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (paramId.getTag() == Parameter::Id::commonTag) {
            _aidl_return->set<Parameter::common>(common_);
            return ndk::ScopedAStatus::ok();
        }
        if (paramId.getTag() == Parameter::Id::loudnessEnhancerTag) {
            // LoudnessEnhancer is an AIDL *union* (vendor | gainMb), not a
            // parcelable with a plain field: the value goes in through set<>().
            LoudnessEnhancer le;
            le.set<LoudnessEnhancer::Tag::gainMb>(encoded_gain_mb());
            Parameter::Specific spec;
            spec.set<Parameter::Specific::loudnessEnhancer>(le);
            _aidl_return->set<Parameter::specific>(spec);
            return ndk::ScopedAStatus::ok();
        }
        return ndk::ScopedAStatus::fromExceptionCode(EX_UNSUPPORTED_OPERATION);
    }

    ndk::ScopedAStatus reopen(OpenEffectReturn* _aidl_return) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.load() == State::INIT) {
            return ndk::ScopedAStatus::fromExceptionCode(EX_ILLEGAL_STATE);
        }
        _aidl_return->statusMQ = status_mq_.dup_descriptor();
        _aidl_return->inputDataMQ = in_mq_.dup_descriptor();
        _aidl_return->outputDataMQ = out_mq_.dup_descriptor();
        return ndk::ScopedAStatus::ok();
    }

private:
    AudioUuid impl_uuid_;
    std::mutex mutex_;
    std::atomic<State> state_{State::INIT};
    std::atomic<ally::Profile> profile_{ally::Profile::Game};
    std::atomic<ally::DeviceKind> device_{ally::DeviceKind::Headphones};
    std::atomic<bool> neural_{false};
    std::atomic<bool> reverb_{true};
    std::atomic<bool> worker_running_{false};
    std::thread worker_;

    Parameter::Common common_{};
    float sample_rate_ = 48000.0f;
    int in_channels_ = 2;
    int out_channels_ = 2;
    size_t frame_count_ = 256;

    ally::Renderer renderer_;
    ally::FmqQueue<IEffect::Status> status_mq_;
    ally::FmqQueue<float> in_mq_;
    ally::FmqQueue<float> out_mq_;

    int32_t encoded_gain_mb() const {
        return 10000 + static_cast<int>(profile_.load()) * 100
             + (neural_.load() ? 10 : 0)
             + static_cast<int>(device_.load());
    }

    void apply_specific_locked(const Parameter::Specific& spec) {
        if (spec.getTag() == Parameter::Specific::loudnessEnhancer) {
            const auto& le = spec.get<Parameter::Specific::loudnessEnhancer>();
            if (le.getTag() != LoudnessEnhancer::Tag::gainMb) {
                return;
            }
            int32_t val = le.get<LoudnessEnhancer::Tag::gainMb>();
            if (val >= 20000 && val < 30000) {
                // profile * 1000 + neural * 100 + reverb * 10 + device
                const int rem = val - 20000;
                const int p = rem / 1000;
                const int n = (rem / 100) % 10;
                const int r = (rem / 10) % 10;
                const int d = rem % 10;
                if (p >= 0 && p <= 4) {
                    auto prof = static_cast<ally::Profile>(p);
                    profile_.store(prof);
                    renderer_.set_profile(prof);
                }
                neural_.store(n != 0);
                renderer_.set_neural(n != 0);
                reverb_.store(r != 0);
                renderer_.set_reverb(r != 0);
                if (d >= 0 && d <= 2) {
                    auto dev = static_cast<ally::DeviceKind>(d);
                    device_.store(dev);
                    renderer_.set_device(dev);
                }
                ALOGI("Updated params: profile=%d neural=%d reverb=%d device=%d", p, n, r, d);
            } else if (val >= 10000 && val < 11000) {
                // Older encoding without the reverb digit, kept so an app that still
                // writes the four-digit code does not lose its settings.
                int rem = val - 10000;
                int p = rem / 100;
                int n = (rem / 10) % 10;
                int d = rem % 10;
                if (p >= 0 && p <= 4) {
                    auto prof = static_cast<ally::Profile>(p);
                    profile_.store(prof);
                    renderer_.set_profile(prof);
                }
                neural_.store(n != 0);
                renderer_.set_neural(n != 0);
                if (d >= 0 && d <= 2) {
                    auto dev = static_cast<ally::DeviceKind>(d);
                    device_.store(dev);
                    renderer_.set_device(dev);
                }
                ALOGI("Updated params: profile=%d neural=%d device=%d", p, n, d);
            }
        }
    }

    void start_worker_locked() {
        worker_running_.store(true);
        worker_ = std::thread([this]() { worker_loop(); });
    }

    void stop_worker() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_worker_locked();
    }

    void stop_worker_locked() {
        if (worker_running_.exchange(false)) {
            in_mq_.wake(0x02);
            if (worker_.joinable()) {
                worker_.join();
            }
        }
    }

    void worker_loop() {
        std::vector<float> in_buf;
        std::vector<float> stereo_buf;
        std::vector<float> out_buf;

        uint64_t stat_buffers = 0, stat_timeouts = 0, stat_wait_us = 0, stat_proc_us = 0;
        int idle_rounds_ = 0;
        auto last_stat = std::chrono::steady_clock::now();
        while (worker_running_.load()) {
            auto t_wait0 = std::chrono::steady_clock::now();
            // Poll at 1 ms while audio flows, back off to 10 ms once the queue has
            // been empty for a few rounds: an idle effect must not wake the mixer
            // thread a thousand times a second.
            const int wait_ms = idle_rounds_ < 3 ? 1 : 10;
            if (!in_mq_.wait(1, wait_ms)) {
                stat_timeouts++;
                if (idle_rounds_ < 1000) idle_rounds_++;
            } else {
                idle_rounds_ = 0;
            }
            auto t_proc0 = std::chrono::steady_clock::now();
            if (!worker_running_.load()) break;

            // Process exactly one buffer per status message: that is what the HAL
            // wrote, and a short or oversized reply makes it wait for the missing
            // frames, which is audible as a hiccup.
            static auto last_beat = std::chrono::steady_clock::now();
            auto now_beat = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now_beat - last_beat).count() >= 2000) {
                ALOGI("effect beat: avail_in=%d avail_out=%d frame_count=%zu state=%d",
                      (int)in_mq_.available_to_read(), (int)out_mq_.available_to_write(),
                      frame_count_, (int)state_.load());
                last_beat = now_beat;
            }
            const size_t want_frames = frame_count_ > 0 ? frame_count_ : 1024;
            const size_t need_in = want_frames * static_cast<size_t>(in_channels_);
            const size_t need_out = want_frames * static_cast<size_t>(out_channels_);
            if (static_cast<size_t>(in_mq_.available_to_read()) < need_in ||
                static_cast<size_t>(out_mq_.available_to_write()) < need_out) {
                continue;
            }
            size_t frames = want_frames;

            in_buf.resize(frames * in_channels_);
            stereo_buf.resize(frames * 2);
            out_buf.resize(frames * out_channels_);

            if (!in_mq_.read(in_buf.data(), frames * in_channels_)) {
                continue;
            }

            if (state_.load() == State::PROCESSING) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (in_channels_ == 1) {
                    // Mono -> duplicate to stereo for renderer
                    std::vector<float> st_in(frames * 2);
                    for (size_t i = 0; i < frames; ++i) {
                        st_in[2 * i] = in_buf[i];
                        st_in[2 * i + 1] = in_buf[i];
                    }
                    renderer_.process_interleaved(st_in.data(), stereo_buf.data(), static_cast<int>(frames), 2);
                } else {
                    renderer_.process_interleaved(in_buf.data(), stereo_buf.data(), static_cast<int>(frames), in_channels_);
                }
            } else {
                // Idle / bypass
                for (size_t i = 0; i < frames; ++i) {
                    stereo_buf[2 * i + 0] = in_buf[i * in_channels_ + 0];
                    stereo_buf[2 * i + 1] = (in_channels_ > 1) ? in_buf[i * in_channels_ + 1] : in_buf[i * in_channels_ + 0];
                }
            }

            if (out_channels_ == 2) {
                out_mq_.write(stereo_buf.data(), frames * 2);
            } else {
                std::fill(out_buf.begin(), out_buf.end(), 0.0f);
                for (size_t i = 0; i < frames; ++i) {
                    out_buf[i * out_channels_ + 0] = stereo_buf[2 * i + 0];
                    if (out_channels_ > 1) out_buf[i * out_channels_ + 1] = stereo_buf[2 * i + 1];
                }
                out_mq_.write(out_buf.data(), frames * out_channels_);
            }

            auto t_end = std::chrono::steady_clock::now();
            stat_buffers++;
            stat_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t_proc0 - t_wait0).count();
            stat_proc_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_proc0).count();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(t_end - last_stat).count() >= 2000) {
                ALOGI("effect stats: buffers=%llu timeouts=%llu avg_wait=%lluus avg_proc=%lluus "
                      "state=%d reverb=%d npu=%d",
                      (unsigned long long)stat_buffers, (unsigned long long)stat_timeouts,
                      (unsigned long long)(stat_buffers ? stat_wait_us / stat_buffers : 0),
                      (unsigned long long)(stat_buffers ? stat_proc_us / stat_buffers : 0),
                      (int)state_.load(), (int)reverb_.load(), (int)renderer_.npu_active());
                stat_buffers = stat_timeouts = stat_wait_us = stat_proc_us = 0;
                last_stat = t_end;
            }

            IEffect::Status st{};
            st.status = 0; // STATUS_OK
            st.fmqConsumed = static_cast<int32_t>(frames * in_channels_);
            st.fmqProduced = static_cast<int32_t>(frames * out_channels_);
            status_mq_.write(&st, 1);
            // libfmq's statusQ->readBlocking() (what AudioFlinger calls) waits on
            // FMQ_NOT_EMPTY == 0x02; waking bit 1 left it blocked forever, which is the
            // frozen playback the device showed.
            status_mq_.wake(0x02);
        }
    }
};

std::mutex g_instances_mutex;
std::vector<std::shared_ptr<AllySoundEffect>> g_instances;

} // namespace
} // namespace aidl::android::hardware::audio::effect

extern "C" binder_status_t queryEffect(
    const aidl::android::media::audio::common::AudioUuid* in_impl_uuid,
    aidl::android::hardware::audio::effect::Descriptor* _aidl_return) {
    if (!in_impl_uuid || !_aidl_return) return STATUS_BAD_VALUE;
    if (!aidl::android::hardware::audio::effect::is_known_uuid(*in_impl_uuid)) return STATUS_NAME_NOT_FOUND;
    *_aidl_return = aidl::android::hardware::audio::effect::make_descriptor(*in_impl_uuid);
    return STATUS_OK;
}

extern "C" binder_status_t createEffect(
    const aidl::android::media::audio::common::AudioUuid* in_impl_uuid,
    std::shared_ptr<aidl::android::hardware::audio::effect::IEffect>* _aidl_return) {
    if (!in_impl_uuid || !_aidl_return) return STATUS_BAD_VALUE;
    if (!aidl::android::hardware::audio::effect::is_known_uuid(*in_impl_uuid)) return STATUS_NAME_NOT_FOUND;
    auto inst = ndk::SharedRefBase::make<aidl::android::hardware::audio::effect::AllySoundEffect>(*in_impl_uuid);
    {
        std::lock_guard<std::mutex> lock(aidl::android::hardware::audio::effect::g_instances_mutex);
        aidl::android::hardware::audio::effect::g_instances.push_back(inst);
    }
    *_aidl_return = inst;
    return STATUS_OK;
}

extern "C" binder_status_t destroyEffect(
    const std::shared_ptr<aidl::android::hardware::audio::effect::IEffect>& in_instance) {
    if (!in_instance) return STATUS_BAD_VALUE;
    in_instance->close();
    std::lock_guard<std::mutex> lock(aidl::android::hardware::audio::effect::g_instances_mutex);
    auto& vec = aidl::android::hardware::audio::effect::g_instances;
    vec.erase(std::remove(vec.begin(), vec.end(), in_instance), vec.end());
    return STATUS_OK;
}
