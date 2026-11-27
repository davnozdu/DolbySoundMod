#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ally {

// Asynchronous StemgenRT (vocals.onnx) vocal separator with optional NNAPI (NPU) support.
// Uses dlopen("libonnxruntime.so") at runtime so the base DSP has zero hard
// dependency on ONNX Runtime when the neural toggle is off.
class PipelinedVocals {
public:
    static constexpr int kHop = 128;
    // The pipeline delay has to cover a whole block that AudioFlinger hands over at
    // once (1920 frames = 15 hops on a deep buffer) plus time for the worker to
    // chew through it. It grows to fit the largest block seen, up to kMaxDelayHops.
    static constexpr int kMinDelayHops = 8;
    static constexpr int kMarginHops = 6;
    static constexpr int kMaxDelayHops = 64;
    // After an underrun stay on the passive matrix for this long before fading the
    // model back in, so a borderline-slow device does not toggle at the block rate.
    static constexpr int kMissCooldownHops = 188; // ~0.5 s at 48 kHz
    // Silence handling, as in the Ally plugin: once the pipeline has drained the
    // model stops running, and after this long without sound it is unloaded.
    static constexpr float kSilenceRms = 1e-5f;      // -100 dBFS
    static constexpr int kUnloadHops = 750;          // ~2 s at 48 kHz

    PipelinedVocals() = default;
    ~PipelinedVocals();

    PipelinedVocals(const PipelinedVocals&) = delete;
    PipelinedVocals& operator=(const PipelinedVocals&) = delete;

    // Non-blocking lazy init: the first call starts loading libonnxruntime.so and the
    // model on a background thread and returns false; it returns true once ready.
    bool ensure_loaded(const char* model_path, bool use_npu = true);
    bool is_ready() const { return ready_.load(); }
    bool using_npu() const { return is_ready() && npu_active_.load(); }
    bool failed() const { return failed_.load(); }

    // Tells the pipeline how many frames the host delivers per call so the delay
    // can cover a whole block. Call from the audio thread before process().
    void note_block_frames(int frames);

    // Call once per hop with the stereo input (planar, 2 * kHop) before
    // ensure_loaded()/process(): tracks silence, gates inference, unloads when idle.
    void note_hop(const float* stereo_in);
    // Stop inference and release the model on its worker, including a load in progress.
    void release_for_idle();

    // Non-blocking process: feeds one stereo hop (planar: 2 * kHop) and outputs
    // time-aligned dry (2 * kHop) and rest = dry - vocals (2 * kHop).
    // Falls back smoothly to passive mid-side matrix on underrun.
    void process(const float* stereo_in, float* dry_out, float* rest_out, uint64_t tick);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::atomic<bool> ready_{false};
    std::atomic<bool> failed_{false};
    std::atomic<bool> loading_{false};
    std::atomic<bool> npu_active_{false};
    std::thread thread_;
    std::mutex lifecycle_mutex_;
    bool stop_requested_ = false; // protected by lifecycle_mutex_

    // Audio-thread only.
    int delay_hops_ = kMinDelayHops;
    bool delay_changed_ = false;
    // Delay line for the dry/passive path (audio thread only), alive across model
    // loads and unloads.
    std::vector<float> delay_buf_;
    size_t delay_pos_ = 0;
    size_t delay_ring_ = 0;
    uint64_t delay_tick_ = UINT64_MAX;
    int silent_hops_ = 0;
    bool previous_output_wet_ = false;
    std::array<float, 2> dry_tail_{}, rest_tail_{};

    void run(const std::string& model_path, bool use_npu);
    void worker_loop(Impl& impl);
    static void passive_matrix(const float* dry, float* rest);
};

} // namespace ally
