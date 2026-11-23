#pragma once

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
    static constexpr int kUnloadHops = 3750;         // ~10 s at 48 kHz

    PipelinedVocals() = default;
    ~PipelinedVocals();

    PipelinedVocals(const PipelinedVocals&) = delete;
    PipelinedVocals& operator=(const PipelinedVocals&) = delete;

    // Non-blocking lazy init: the first call starts loading libonnxruntime.so and the
    // model on a background thread and returns false; it returns true once ready.
    bool ensure_loaded(const char* model_path, bool use_npu = true);
    bool is_ready() const { return ready_.load(); }
    bool using_npu() const { return npu_active_.load(); }

    // Tells the pipeline how many frames the host delivers per call so the delay
    // can cover a whole block. Call from the audio thread before process().
    void note_block_frames(int frames);

    // Call once per hop with the stereo input (planar, 2 * kHop) before
    // ensure_loaded()/process(): tracks silence, gates inference, unloads when idle.
    void note_hop(const float* stereo_in);
    // The host delivered no audio at all for `frames` frames (stream stopped).
    void idle(int frames);

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
    std::thread loader_;
    std::string model_path_;

    // Audio-thread only.
    int delay_hops_ = kMinDelayHops;
    bool delay_changed_ = false;
    int silent_hops_ = 0;

    void unload();
    void teardown();

    void load(bool use_npu);
    void worker_loop();
    static void passive_matrix(const float* dry, float* rest);
};

} // namespace ally
