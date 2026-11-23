#include "auto_eq.hpp"
#include "qnn_runtime.hpp"
#include "spatial.hpp"

#include <onnxruntime_c_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <mutex>
#include <numeric>
#include <thread>
#include <unistd.h>
#ifdef __ANDROID__
#include <android/log.h>
#include <sched.h>
#include <sys/resource.h>
#else
enum { ANDROID_LOG_DEBUG = 3, ANDROID_LOG_INFO = 4, ANDROID_LOG_WARN = 5 };
#endif

namespace ally {
namespace {
constexpr int kRate = 16000;
constexpr int kPatch = 15600;
// Discogs-EffNet (Essentia, 2022): MusiCNN-style mel input, 128-frame patches, 400 Discogs
// styles. Three patches (hop 56 frames) cover the last ~3.9 s and are averaged.
constexpr int kGenrePatchRows = 128;
constexpr int kGenrePatches = 3;
constexpr int kGenrePatchHop = 56;
constexpr int kGenreRows = kGenrePatchRows + (kGenrePatches - 1) * kGenrePatchHop;
constexpr int kGenreSamples = 512 + (kGenreRows - 1) * 256;
constexpr int kStyles = 400;
// Decision thresholds on the averaged sigmoid activations (calibrated on labeled clips).
constexpr float kGenreMinimum = 0.15f;
constexpr float kGenreMargin = 1.25f;
// Discogs style -> EQ group, in the model's class order (its metadata "classes"; the
// model is pinned by SHA-256 in tools/fetch_auto_eq_models.sh): 0 other music, 1 metal,
// 2 rock, 3 jazz, 4 classical/score/soundtrack, 5 electronic, 6 hip hop.
constexpr char kStyleGroups[] =
    "0000000000000000004444444444444555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555500000000000000000000000000000000000000000066666666666666666666666666333333333333333333333333300000000000000000000000000000000000000000000000000000000000000000000000000022222212212222221121212222121122121122212222221212222221222222212122212222222212221222112124440";
static_assert(sizeof(kStyleGroups) == kStyles + 1, "one group per Discogs style");
constexpr double kPi = 3.14159265358979323846;
// Battery: classify in short probes, not every window. A probe lasts until the vote
// is stable (two equal results, plus one MusiCNN pass for music: ~2-5 windows of
// ~1 s), then the classifier rests for kRestWindows windows (~20 s) before checking
// the mix again. A changed result inside a probe simply extends the probe.
constexpr int kRestWindows = 20;
// Release the ONNX sessions (and their HTP context) after this much silence or no audio.
constexpr float kIdleReleaseSeconds = 10.0f;
#ifndef ALLY_EQ_MODEL_DIR
#define ALLY_EQ_MODEL_DIR "/vendor/lib64/soundfx/"
#endif

enum class Eq { Neutral, Speech, Music, Metal, Rock, Jazz, Classical, Electronic, HipHop, Error };
constexpr const char* kNames[] = {
    "neutral", "speech", "music", "metal", "rock", "jazz", "classical", "electronic", "hiphop", "error"
};
constexpr float kCurves[][3] = {
    {0, 0, 0}, {-1.8f, 2, 0}, {0, 0, 0}, {0.5f, 0.2f, -1}, {0.5f, 0.8f, -0.2f},
    {0, -0.4f, 0.8f}, {0, 0, 0.5f}, {1, -0.4f, 0.3f}, {1.2f, 0.4f, 0}, {0, 0, 0}
};

class Models {
public:
    ~Models() {
        if (genre_) api_->ReleaseSession(genre_);
        if (audio_) api_->ReleaseSession(audio_);
        if (qnn_audio_) {
            const std::lock_guard<std::mutex> lock(qnn_mutex());
            api_->ReleaseSession(qnn_audio_);
        }
        if (qnn_options_) api_->ReleaseSessionOptions(qnn_options_);
        if (memory_) api_->ReleaseMemoryInfo(memory_);
        if (options_) api_->ReleaseSessionOptions(options_);
        if (env_) api_->ReleaseEnv(env_);
        if (lib_) dlclose(lib_);
    }

    bool open() {
        constexpr const char* qnn = ALLY_EQ_MODEL_DIR "ally-qnn/libonnxruntime.so";
        if (access(qnn, R_OK) == 0) lib_ = dlopen(qnn, RTLD_NOW | RTLD_LOCAL);
        const bool qnn_runtime_loaded = lib_ != nullptr;
        if (!lib_) lib_ = dlopen(ALLY_EQ_MODEL_DIR "libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) lib_ = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) return false;
        auto base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(lib_, "OrtGetApiBase"));
        if (!base || !(api_ = base()->GetApi(ORT_API_VERSION))) return false;
        const bool ready = ok(api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "AllyAutoEq", &env_)) &&
                           ok(api_->CreateSessionOptions(&options_)) &&
                           ok(api_->SetIntraOpNumThreads(options_, 1)) &&
                           ok(api_->SetInterOpNumThreads(options_, 1)) &&
                           ok(api_->AddSessionConfigEntry(options_, "session.intra_op.allow_spinning", "0")) &&
                           ok(api_->DisableCpuMemArena(options_)) &&
                           ok(api_->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_)) &&
                           ok(api_->CreateSession(env_, ALLY_EQ_MODEL_DIR "yamnet.onnx", options_, &audio_));
        if (ready && qnn_runtime_loaded) enable_qnn();
        return ready;
    }

    template <size_t D, size_t N>
    bool run(bool genre, std::vector<float>& features, const int64_t (&shape)[D],
             std::array<float, N>& scores) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (genre && !genre_ && !open_genre()) return false;
        const char* input_name = genre ? "melspectrogram" : "audio";
        const char* output_name = genre ? "activations" : "class_scores";
        OrtValue* input = nullptr;
        if (!ok(api_->CreateTensorWithDataAsOrtValue(
                memory_, features.data(), features.size() * sizeof(float), shape, D,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input))) {
            return false;
        }
        // Genre stays on the CPU: measured on the phone, the fp16 HTP result ran up to
        // 0.09 below the CPU's group scores (rock 0.20 -> 0.16 against a 0.15 threshold),
        // while the model runs about once per 25 s, so the HTP would only cost accuracy.
        OrtSession* npu = genre ? nullptr : (qnn_healthy_ ? qnn_audio_ : nullptr);
        bool done = false;
        if (npu) {
            // Another model holding the HTP is not worth waiting for: use the CPU.
            std::unique_lock<std::mutex> qnn_lock(qnn_mutex(), std::try_to_lock);
            if (qnn_lock) {
                const auto start = std::chrono::steady_clock::now();
                const bool ran = infer(npu, input_name, input, output_name, scores);
                const float ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                const char* model = "YAMNet";
                if (!ran) {
                    // Background thread with seconds of slack: a slow run is fine, only a
                    // failure moves the model to the CPU for good.
                    qnn_healthy_ = false;
                    log(ANDROID_LOG_WARN, "%s HTP failed after %.2f ms, using CPU", model, ms);
                } else {
                    done = true;
                    log(ANDROID_LOG_DEBUG, "%s HTP %.2f ms", model, ms);
                }
            }
        }
        if (!done) done = infer(genre ? genre_ : audio_, input_name, input, output_name, scores);
        api_->ReleaseValue(input);
        return done;
    }

private:
    __attribute__((format(printf, 2, 3)))
    static void log(int priority, const char* format, ...) {
#ifdef __ANDROID__
        va_list args;
        va_start(args, format);
        __android_log_vprint(priority, "AllyAutoEq", format, args);
        va_end(args);
#else
        (void)priority, (void)format;
#endif
    }

    // tools/prepare_effnet.py fixes the batch to kGenrePatches and drops the unused
    // embeddings output, whose Identity the HTP rejects.
    bool open_genre() {
        return ok(api_->CreateSession(env_, ALLY_EQ_MODEL_DIR "discogs-effnet-htp.onnx", options_, &genre_));
    }

    void enable_qnn() {
        if (!ok(api_->CloneSessionOptions(options_, &qnn_options_))) return;
        const char* keys[] = {"backend_path", "enable_htp_fp16_precision",
                              "htp_performance_mode", "qnn_context_priority"};
        const char* values[] = {ALLY_EQ_MODEL_DIR "ally-qnn/libQnnHtp.so", "1", kQnnBackgroundMode, "low"};
        const bool provider = ok(api_->AddSessionConfigEntry(qnn_options_, "session.disable_cpu_ep_fallback", "1")) &&
                              with_qnn_library_path(ALLY_EQ_MODEL_DIR "ally-qnn/", [&] {
                                  return ok(api_->SessionOptionsAppendExecutionProvider(qnn_options_, "QNN", keys,
                                                                                         values, 4));
                              });
        if (!provider) {
            api_->ReleaseSessionOptions(qnn_options_);
            qnn_options_ = nullptr;
            return;
        }
        qnn_healthy_ = with_qnn_library_path(ALLY_EQ_MODEL_DIR "ally-qnn/", [&] {
            return create_cached_qnn_session(api_, env_, qnn_options_, ALLY_EQ_MODEL_DIR "yamnet.onnx",
                                             "yamnet_ctx.onnx", &qnn_audio_,
                                             [this](OrtStatus* st) { return ok(st); });
        });
        // Only YAMNet uses the HTP; its session keeps what it needs.
        api_->ReleaseSessionOptions(qnn_options_);
        qnn_options_ = nullptr;
        if (qnn_healthy_) log(ANDROID_LOG_INFO, "YAMNet HTP ready");
    }

    template <size_t N>
    bool infer(OrtSession* session, const char* input_name, OrtValue* input,
               const char* output_name, std::array<float, N>& scores) {
        OrtValue* output = nullptr;
        float* data = nullptr;
        bool done = ok(api_->Run(session, nullptr, &input_name, &input, 1, &output_name, 1, &output)) &&
                    ok(api_->GetTensorMutableData(output, reinterpret_cast<void**>(&data)));
        if (done) {
            std::copy_n(data, N, scores.begin());
            done = std::all_of(scores.begin(), scores.end(), [](float x) { return std::isfinite(x); });
        }
        if (output) api_->ReleaseValue(output);
        return done;
    }

    bool ok(OrtStatus* status) const {
        if (!status) return true;
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "AllyAutoEq", "%s", api_->GetErrorMessage(status));
#endif
        api_->ReleaseStatus(status);
        return false;
    }
    void* lib_ = nullptr;
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* options_ = nullptr;
    OrtMemoryInfo* memory_ = nullptr;
    OrtSession* audio_ = nullptr;
    OrtSession* genre_ = nullptr;
    OrtSession* qnn_audio_ = nullptr;
    OrtSessionOptions* qnn_options_ = nullptr;   // CPU options plus the QNN provider
    bool qnn_healthy_ = false;
    std::mutex mutex_;
};

std::shared_ptr<Models> acquire_models() {
    static std::mutex mutex;
    static std::weak_ptr<Models> shared;
    const std::lock_guard<std::mutex> lock(mutex);
    if (auto active = shared.lock()) return active;
    auto opened = std::make_shared<Models>();
    if (!opened->open()) return {};
    shared = opened;
    return opened;
}

// Both models use 512-point FFTs, with different windows and mel filter banks.
struct Features {
    explicit Features(bool genre)
        : frame_size(genre ? 512 : 400), step(genre ? 256 : 160),
          rows(genre ? kGenreRows : 96), bands(genre ? 96 : 64),
          mel(257 * bands), output(rows * bands) {
        for (int i = 0; i < frame_size; ++i) {
            window[i] = 0.5 - 0.5 * std::cos(2 * kPi * i / (genre ? frame_size - 1 : frame_size));
        }
        const double log_step = std::log(6.4) / 27;
        auto htk = [](double hz) { return 1127 * std::log1p(hz / 700); };
        auto edge = [&](int index) {
            const double fraction = static_cast<double>(index) / (bands + 1);
            if (!genre) return htk(125) + fraction * (htk(7500) - htk(125));
            const double point = fraction * (15 + std::log(8.0) / log_step);
            return point <= 15 ? point * 1000 / 15 : 1000 * std::exp(log_step * (point - 15));
        };
        for (int bin = 0; bin <= 256; ++bin) {
            const double hz = bin * (kRate / 512.0);
            const double x = genre ? hz : htk(hz);
            for (int band = 0; band < bands; ++band) {
                const double low = edge(band), mid = edge(band + 1), high = edge(band + 2);
                const double value = std::max(0.0, std::min((x - low) / (mid - low), (high - x) / (high - mid)));
                mel[bin * bands + band] = genre ? 2 * value / (high - low) : value;
            }
        }
    }

    void compute(const float* audio, bool genre) {
        for (int row = 0; row < rows; ++row) {
            for (int i = 0; i < frame_size; ++i) frame[i] = window[i] * audio[row * step + i];
            fft.forward(frame.data(), spectrum.data());
            auto* dest = output.data() + row * bands;
            std::fill_n(dest, bands, 0.0f);
            for (int bin = 0; bin <= 256; ++bin) {
                const float power = std::norm(spectrum[bin]);
                const float weight = genre ? power : std::sqrt(power);
                for (int band = 0; band < bands; ++band) dest[band] += weight * mel[bin * bands + band];
            }
            for (int band = 0; band < bands; ++band) {
                dest[band] = genre ? std::log10(1.0f + 10000 * dest[band]) : std::log(dest[band] + 0.001f);
            }
        }
    }

    int frame_size, step, rows, bands;
    RealFFT fft{512};
    std::array<float, 512> window{}, frame{};
    std::array<std::complex<float>, 257> spectrum{};
    std::vector<float> mel, output;
};

class Resampler {
public:
    explicit Resampler(float rate) : ratio_(rate / kRate) {
        // A short windowed-sinc anti-aliasing filter with fractional phases.
        const double cutoff = 0.48 / std::max(1.0, ratio_);
        for (size_t phase = 0; phase < kernel_.size(); ++phase) {
            auto& taps = kernel_[phase];
            for (int tap = 0; tap < 64; ++tap) {
                const double x = tap - 31 - static_cast<double>(phase) / kernel_.size();
                const double angle = 2 * kPi * cutoff * x;
                const double sinc = angle == 0 ? 1 : std::sin(angle) / angle;
                taps[tap] = 2 * cutoff * sinc * (0.5 + 0.5 * std::cos(kPi * x / 33));
            }
            const float sum = std::accumulate(taps.begin(), taps.end(), 0.0f);
            for (float& tap : taps) tap /= sum;
        }
    }

    std::vector<float> convert(const std::vector<float>& audio) const {
        std::vector<float> mono(kPatch);
        for (int i = 0; i < kPatch; ++i) {
            const double position = i * ratio_;
            const int base = static_cast<int>(position);
            const auto& taps = kernel_[static_cast<size_t>((position - base) * kernel_.size())];
            for (int tap = 0; tap < 64; ++tap) {
                const int sample = base + tap - 31;
                if (sample >= 0 && sample < static_cast<int>(audio.size())) {
                    mono[i] += audio[sample] * taps[tap];
                }
            }
        }
        return mono;
    }

private:
    double ratio_;
    std::array<std::array<float, 64>, 64> kernel_{};
};

template <size_t N>
float peak(const std::array<float, N>& scores, std::initializer_list<int> labels) {
    float best = 0.0f;
    for (int label : labels) best = std::max(best, scores[label]);
    return best;
}

struct Tag { Eq profile; float score; };

template <size_t N>
Tag best_tag(std::array<Tag, N>& tags) {
    std::sort(tags.begin(), tags.end(), [](Tag a, Tag b) { return a.score > b.score; });
    return tags[0];
}

Eq audio_profile(std::array<float, 521>& scores) {
    for (float& score : scores) score = 1.0f / (1.0f + std::exp(-score));
    const float speech = peak(scores, {0, 1, 2, 3, 5});
    std::array<Tag, 6> tags{{
        {Eq::Metal, scores[215]}, {Eq::Rock, peak(scores, {214, 216, 217, 218, 219, 220})},
        {Eq::Jazz, peak(scores, {225, 230})}, {Eq::Classical, peak(scores, {232, 233})},
        {Eq::Electronic, peak(scores, {234, 235, 236, 237, 238, 239, 240, 241, 242})},
        {Eq::HipHop, scores[212]},
    }};
    const auto top = best_tag(tags);
    const float music = std::max(top.score, peak(scores, {24, 132}));
    if (speech >= 0.55f && speech > music + 0.14f) return Eq::Speech;
    if (music < 0.50f) return Eq::Neutral;
    return top.score >= 0.42f && top.score > tags[1].score + 0.07f && speech < 0.55f
           ? top.profile : Eq::Music;
}

using Groups = std::array<float, 7>;

// Average the patches, then score each EQ group by its strongest style.
Groups style_groups(const std::array<float, kGenrePatches * kStyles>& scores) {
    Groups group{};
    for (int style = 0; style < kStyles; ++style) {
        float mean = 0.0f;
        for (int patch = 0; patch < kGenrePatches; ++patch) mean += scores[patch * kStyles + style];
        mean /= kGenrePatches;
        const int g = kStyleGroups[style] - '0';
        group[g] = std::max(group[g], mean);
    }
    return group;
}

// Decides on the group scores averaged over every genre pass of the current probe, so
// one ambiguous window cannot flip the result.
Eq genre_profile(const Groups& group, Eq fallback) {
#ifndef __ANDROID__
    if (std::getenv("ALLY_EQ_DEBUG")) {
        std::fprintf(stderr, "genre groups: other %.2f metal %.2f rock %.2f jazz %.2f classical %.2f "
                     "electronic %.2f hiphop %.2f\n", group[0], group[1], group[2], group[3], group[4],
                     group[5], group[6]);
    }
#endif
    std::array<Tag, 6> tags{{
        {Eq::Metal, group[1]}, {Eq::Rock, group[2]}, {Eq::Jazz, group[3]},
        {Eq::Classical, group[4]}, {Eq::Electronic, group[5]}, {Eq::HipHop, group[6]},
    }};
    const auto top = best_tag(tags);
    // Metal styles also light up rock; let a clear metal score win over its parent.
    if (top.profile == Eq::Rock && group[1] >= kGenreMinimum && group[1] >= 0.6f * group[2]) return Eq::Metal;
    const float competitor = std::max(tags[1].score, group[0]);
    return top.score >= kGenreMinimum && top.score > competitor * kGenreMargin ? top.profile : fallback;
}

} // namespace

struct AutoEq::Impl {
    explicit Impl(float sample_rate)
        : rate(sample_rate), capture(std::ceil(kPatch * rate / kRate)), job(capture.size()),
          slew(1 - std::exp(-kHop / (rate * 0.65f))) {
        low.configure_lowpass_2nd(180, rate, 2);
        mid_hp.configure_highpass_2nd(1000, rate, 2);
        mid_lp.configure_lowpass_2nd(4200, rate, 2);
        high.configure_highpass_2nd(5200, rate, 2);
        for (size_t p = 0; p < targets.size(); ++p) {
            for (size_t band = 0; band < gains.size(); ++band) {
                targets[p][band] = std::pow(10.0f, kCurves[p][band] / 20) - 1;
            }
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv.notify_one();
        if (worker.joinable()) worker.join();
    }

    Eq profile() const {
        const auto value = result.load(std::memory_order_relaxed);
        return (value >> 8) == generation ? static_cast<Eq>(value & 255) : Eq::Neutral;
    }

    void invalidate() {
        ++generation;
        filled = 0;
        rest.store(0, std::memory_order_relaxed);
    }

    void request_release() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            release = true;
        }
        cv.notify_one();
    }

    void reset_filter() {
        low.reset();
        mid_hp.reset();
        mid_lp.reset();
        high.reset();
        gains.fill(0);
    }

    void run() {
#ifdef __ANDROID__
        sched_param priority{};
        sched_setscheduler(0, SCHED_OTHER, &priority);
        setpriority(PRIO_PROCESS, 0, 5);
#endif
        Resampler resampler(rate);
        Features yamnet(false), musicnn(true);  // musicnn: the MusiCNN-style mel input EffNet uses
        std::shared_ptr<Models> models;
        std::vector<float> window(std::ceil(kPatch * rate / kRate));
        std::deque<std::vector<float>> music;
        std::array<float, kGenreSamples> genre_audio{};
        std::vector<float> genre_batch(kGenrePatches * kGenrePatchRows * 96);
        uint64_t previous = 0;
        Eq vote = Eq::Neutral;
        int count = 0;
        int silent_windows = 0;
        bool genre_ran = false;
        Groups group_sum{};
        int group_runs = 0;
        const int idle_windows = static_cast<int>(kIdleReleaseSeconds * rate / window.size()) + 1;
        while (true) {
            uint64_t token;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stopping || pending || release; });
                if (stopping) return;
                if (release) {
                    // Paused, stopped or switched off: give the memory back.
                    release = false;
                    models.reset();
                    music.clear();
                    silent_windows = 0;
                    if (!pending) continue;
                }
                pending = false;
                token = job_generation;
                window.swap(job);
            }
            if (token != previous) {
                previous = token;
                music.clear();
                vote = Eq::Neutral;
                count = 0;
                genre_ran = false;
                group_sum = {};
                group_runs = 0;
            }
            const float energy = std::inner_product(window.begin(), window.end(), window.begin(), 0.0f);
            const float rms = std::sqrt(energy / window.size());
            Eq candidate = Eq::Neutral;
            const bool audible = std::isfinite(rms) && rms >= 1e-5f;
            silent_windows = audible ? 0 : silent_windows + 1;
            if (!audible && silent_windows >= idle_windows) models.reset();
            if (audible) {
                if (!models) models = acquire_models();
                auto mono = resampler.convert(window);
                // AudioFlinger can attenuate samples before the session effect.
                // Normalize only the analysis copy so playback volume cannot mute classification.
                const float gain = std::clamp(0.04f / rms, 1.0f, 512.0f);
                for (float& sample : mono) sample *= gain;
                yamnet.compute(mono.data(), false);
                std::array<float, 521> audio_scores{};
                const int64_t audio_shape[] = {1, 1, 96, 64};
                if (!models || !models->run(false, yamnet.output, audio_shape, audio_scores)) {
                    candidate = Eq::Error;
                } else {
                    candidate = audio_profile(audio_scores);
                }
                if (candidate >= Eq::Music && candidate < Eq::Error) {
                    if (music.size() == 4) music.pop_front();
                    music.push_back(std::move(mono));
                    if (music.size() == 4) {
                        constexpr int first = 4 * kPatch - kGenreSamples;
                        auto end = std::copy(music.front().begin() + first, music.front().end(), genre_audio.begin());
                        for (size_t i = 1; i < music.size(); ++i) end = std::copy(music[i].begin(), music[i].end(), end);
                        musicnn.compute(genre_audio.data(), true);
                        // Three overlapping 128-frame patches, laid out as one batch.
                        for (int patch = 0; patch < kGenrePatches; ++patch) {
                            std::copy_n(musicnn.output.begin() + patch * kGenrePatchHop * 96,
                                        kGenrePatchRows * 96, genre_batch.begin() + patch * kGenrePatchRows * 96);
                        }
                        std::array<float, kGenrePatches * kStyles> genre_scores{};
                        const int64_t genre_shape[] = {kGenrePatches, kGenrePatchRows, 96};
                        genre_ran = true;
                        if (models->run(true, genre_batch, genre_shape, genre_scores)) {
                            const Groups current = style_groups(genre_scores);
                            for (size_t g = 0; g < group_sum.size(); ++g) group_sum[g] += current[g];
                            ++group_runs;
                            Groups mean{};
                            for (size_t g = 0; g < mean.size(); ++g) mean[g] = group_sum[g] / group_runs;
                            candidate = genre_profile(mean, candidate);
                        } else {
                            candidate = Eq::Error;
                        }
                    }
                } else {
                    music.clear();
                }
            } else {
                music.clear();
            }
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_DEBUG, "AllyAutoEq", "window rms=%.5f candidate=%s",
                                rms, kNames[static_cast<size_t>(candidate)]);
#endif
            count = candidate == vote ? std::min(count + 1, 2) : 1;
            vote = candidate;
            // Music only counts once its genre has been checked in this probe: otherwise
            // the first windows of every probe store plain "music" and flatten the
            // genre curve for a few seconds every probe.
            const bool musical = candidate >= Eq::Music && candidate < Eq::Error;
            if ((count >= 2 && (!musical || genre_ran)) || candidate == Eq::Error) {
                // An unsure probe ("music" without a confident genre) keeps the genre an
                // earlier probe found for this stream; only a confident genre replaces it.
                const uint64_t stored = result.load(std::memory_order_relaxed);
                const Eq previous_profile = (stored >> 8) == token ? static_cast<Eq>(stored & 255) : Eq::Neutral;
                const bool keep_genre = candidate == Eq::Music && previous_profile > Eq::Music &&
                                        previous_profile < Eq::Error;
                if (!keep_genre) {
                    result.store((token << 8) | static_cast<uint64_t>(candidate), std::memory_order_relaxed);
                }
                if (candidate != Eq::Error) {
                    rest.store(kRestWindows, std::memory_order_relaxed);
                    music.clear();          // the next probe needs fresh, contiguous audio
                    genre_ran = false;
                    group_sum = {};
                    group_runs = 0;
                    count = 0;
                }
            }
        }
    }

    float rate;
    std::vector<float> capture, job;
    size_t filled = 0;
    uint64_t generation = 1, job_generation = 0;
    std::atomic<uint64_t> result{0};
    bool on = false, pending = false, stopping = false, release = false;
    std::atomic<int> rest{0};        // windows to skip before the next probe
    float idle_frames = 0.0f;        // audio thread: time without any audio callback
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    BiquadFilter low, mid_hp, mid_lp, high;
    std::array<std::array<float, 3>, std::size(kCurves)> targets{};
    std::array<float, 3> gains{};
    float slew;
};

AutoEq::AutoEq() = default;
AutoEq::~AutoEq() = default;

void AutoEq::init(float rate) {
    if (rate == rate_) return;
    const bool active = enabled();
    impl_.reset();
    rate_ = rate;
    set_enabled(active);
}

void AutoEq::reset() {
    if (!impl_) return;
    impl_->invalidate();
    impl_->reset_filter();
}

void AutoEq::idle(int frames) {
    if (!impl_ || !impl_->on) return;
    auto& state = *impl_;
    const bool was_active = state.idle_frames < kIdleReleaseSeconds * rate_;
    state.idle_frames += static_cast<float>(frames);
    if (was_active && state.idle_frames >= kIdleReleaseSeconds * rate_) state.request_release();
}

void AutoEq::set_enabled(bool enabled) {
    if (enabled && !impl_) impl_ = std::make_unique<Impl>(rate_);
    if (!impl_ || impl_->on == enabled) return;
    impl_->invalidate();
    impl_->on = enabled;
    if (!enabled && impl_->worker.joinable()) impl_->request_release();
    if (enabled && !impl_->worker.joinable()) {
        impl_->worker = std::thread(&Impl::run, impl_.get());
    }
}

bool AutoEq::enabled() const {
    return impl_ && impl_->on;
}

void AutoEq::observe(const float* left, const float* right) {
    if (!enabled() || impl_->profile() == Eq::Error) return;
    auto& state = *impl_;
    state.idle_frames = 0.0f;
    for (int i = 0; i < kHop; ++i) {
        state.capture[state.filled++] = 0.5f * (left[i] + right[i]);
        if (state.filled != state.capture.size()) continue;
        state.filled = 0;
        if (const int r = state.rest.load(std::memory_order_relaxed); r > 0) {
            state.rest.store(r - 1, std::memory_order_relaxed);
            continue;
        }
        std::unique_lock<std::mutex> lock(state.mutex, std::try_to_lock);
        if (!lock || state.pending) continue;
        state.capture.swap(state.job);
        state.job_generation = state.generation;
        state.pending = true;
        lock.unlock();
        state.cv.notify_one();
    }
}

void AutoEq::apply(float* left, float* right) {
    if (!impl_) return;
    auto& state = *impl_;
    const auto& target = state.targets[static_cast<size_t>(state.profile())];
    std::array<float, 3> change{};
    bool flat = true;
    for (size_t band = 0; band < change.size(); ++band) {
        change[band] = (target[band] - state.gains[band]) * state.slew;
        flat &= target[band] == 0 && std::abs(state.gains[band]) < 1e-5f;
    }
    if (flat) {
        if (state.gains != std::array<float, 3>{}) state.reset_filter();
        return;
    }
    std::array<float, kHop> low{}, mid{}, high{};
    float* channels[] = {left, right};
    for (int ch = 0; ch < 2; ++ch) {
        float* audio = channels[ch];
        state.low.process(audio, low.data(), ch, kHop);
        state.mid_hp.process(audio, mid.data(), ch, kHop);
        state.mid_lp.process_inplace(mid.data(), ch, kHop);
        state.high.process(audio, high.data(), ch, kHop);
        for (int i = 0; i < kHop; ++i) {
            const float t = (i + 1.0f) / kHop;
            audio[i] += low[i] * (state.gains[0] + t * change[0]) +
                        mid[i] * (state.gains[1] + t * change[1]) +
                        high[i] * (state.gains[2] + t * change[2]);
        }
    }
    for (size_t band = 0; band < change.size(); ++band) state.gains[band] += change[band];
}

const char* AutoEq::profile() const {
    return kNames[impl_ ? static_cast<size_t>(impl_->profile()) : 0];
}

} // namespace ally
