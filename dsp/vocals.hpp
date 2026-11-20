#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace ally {

// Asynchronous StemgenRT (vocals.onnx) vocal separator with NNAPI (NPU) support.
// Uses dlopen("libonnxruntime.so") at runtime so the base DSP has zero hard
// dependency on ONNX Runtime when the neural toggle is off.
class PipelinedVocals {
public:
    static constexpr int kHop = 128;
    static constexpr int kBufferHops = 10;

    PipelinedVocals() = default;
    ~PipelinedVocals();

    PipelinedVocals(const PipelinedVocals&) = delete;
    PipelinedVocals& operator=(const PipelinedVocals&) = delete;

    // Lazy init: loads libonnxruntime.so and vocals.onnx. Returns true if ready.
    bool ensure_loaded(const char* model_path, bool use_npu = true);
    bool is_ready() const { return ready_.load(); }
    bool using_npu() const { return npu_active_.load(); }

    // Non-blocking process: feeds one stereo hop (planar: 2 * kHop) and outputs
    // time-aligned dry (2 * kHop) and rest = dry - vocals (2 * kHop).
    // Falls back smoothly to passive mid-side matrix on underrun.
    void process(const float* stereo_in, float* dry_out, float* rest_out, uint64_t tick);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::atomic<bool> ready_{false};
    std::atomic<bool> failed_{false};
    std::atomic<bool> npu_active_{false};

    void worker_loop();
    static void passive_matrix(const float* dry, float* rest);
};

} // namespace ally
