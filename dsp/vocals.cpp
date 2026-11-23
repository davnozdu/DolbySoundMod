#include "vocals.hpp"

#include <dlfcn.h>
#include <onnxruntime_c_api.h>
#include <nnapi_provider_factory.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#define VLOGI(...) __android_log_print(ANDROID_LOG_INFO, "AllySoundNPU", __VA_ARGS__)
#define VLOGE(...) __android_log_print(ANDROID_LOG_ERROR, "AllySoundNPU", __VA_ARGS__)
#else
#define VLOGI(...)
#define VLOGE(...)
#endif

namespace ally {

namespace {

constexpr int kHop = PipelinedVocals::kHop;
using HopBuf = std::array<float, 2 * kHop>;

struct QueuedHop {
    uint64_t gen;
    uint64_t seq;
    HopBuf audio;
};

struct OutHop {
    uint64_t gen;
    uint64_t seq;
    HopBuf dry;
    HopBuf rest;
};

// NNAPI without a vendor driver only has "nnapi-reference": a slow CPU emulation
// that, with relaxed fp16, adds audible noise to the recurrent state. Only hand the
// model to NNAPI when a real accelerator is registered.
bool has_nnapi_accelerator() {
    void* nn = dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);
    if (nn == nullptr) {
        VLOGI("libneuralnetworks.so unavailable: %s", dlerror());
        return false;
    }
    using CountFn = int (*)(uint32_t*);
    using GetFn = int (*)(uint32_t, void**);
    using NameFn = int (*)(const void*, const char**);
    using TypeFn = int (*)(const void*, int32_t*);
    auto count_fn = reinterpret_cast<CountFn>(dlsym(nn, "ANeuralNetworks_getDeviceCount"));
    auto get_fn = reinterpret_cast<GetFn>(dlsym(nn, "ANeuralNetworks_getDevice"));
    auto name_fn = reinterpret_cast<NameFn>(dlsym(nn, "ANeuralNetworksDevice_getName"));
    auto type_fn = reinterpret_cast<TypeFn>(dlsym(nn, "ANeuralNetworksDevice_getType"));
    bool found = false;
    uint32_t count = 0;
    if (count_fn && get_fn && name_fn && type_fn && count_fn(&count) == 0) {
        for (uint32_t i = 0; i < count; ++i) {
            void* dev = nullptr;
            const char* name = "?";
            int32_t type = 0;
            if (get_fn(i, &dev) != 0 || dev == nullptr) continue;
            name_fn(dev, &name);
            type_fn(dev, &type);
            // ANEURALNETWORKS_DEVICE_CPU == 2
            const bool real = type != 2 && std::strcmp(name, "nnapi-reference") != 0;
            VLOGI("NNAPI device %u: %s type=%d%s", i, name, type, real ? " (accelerator)" : "");
            found = found || real;
        }
    }
    dlclose(nn);
    return found;
}

// One ONNX Runtime + one compiled session for the whole process. Every effect
// instance (one per app session) only owns its recurrent state; OrtSession::Run is
// thread-safe, so instances share the ~30 MB of runtime, weights and graph.
struct SharedModel {
    void* lib = nullptr;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* opts = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* mem_info = nullptr;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> state_shapes;
    std::vector<size_t> state_sizes;
    bool npu = false;
    int refs = 0;

    ~SharedModel() {
        if (api) {
            if (session) api->ReleaseSession(session);
            if (opts) api->ReleaseSessionOptions(opts);
            if (mem_info) api->ReleaseMemoryInfo(mem_info);
            if (env) api->ReleaseEnv(env);
        }
        if (lib) dlclose(lib);
    }
};

std::mutex g_model_mutex;
SharedModel* g_model = nullptr;

#define ORT_OK(expr) (([&]() { OrtStatus* _s = (expr); if (_s) { VLOGE("%s: %s", #expr, m->api->GetErrorMessage(_s)); m->api->ReleaseStatus(_s); return false; } return true; })())

SharedModel* load_model(const char* model_path, bool use_npu) {
    auto* m = new SharedModel();
    static const char* kOrtCandidates[] = {
        "libonnxruntime.so",
        "/vendor/lib64/soundfx/libonnxruntime.so",
        "/vendor/lib64/libonnxruntime.so",
    };
    for (const char* cand : kOrtCandidates) {
        m->lib = dlopen(cand, RTLD_NOW | RTLD_LOCAL);
        if (m->lib) {
            VLOGI("onnxruntime loaded: %s", cand);
            break;
        }
        VLOGE("dlopen %s failed: %s", cand, dlerror());
    }
    if (!m->lib) { delete m; return nullptr; }

    using GetApiBaseFn = const OrtApiBase* (*)();
    auto get_base = reinterpret_cast<GetApiBaseFn>(dlsym(m->lib, "OrtGetApiBase"));
    if (!get_base || !(m->api = get_base()->GetApi(ORT_API_VERSION))) {
        VLOGE("OrtApi %d unavailable", ORT_API_VERSION);
        delete m;
        return nullptr;
    }
    if (!ORT_OK(m->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "AllySound", &m->env))) {
        delete m;
        return nullptr;
    }

    auto create_session = [&](bool try_npu) -> bool {
        if (m->opts) { m->api->ReleaseSessionOptions(m->opts); m->opts = nullptr; }
        if (!ORT_OK(m->api->CreateSessionOptions(&m->opts))) return false;
        // One hop is 128 samples: a thread pool costs more than it saves, spinning
        // pool threads burn a core, and the arena keeps peak buffers forever.
        ORT_OK(m->api->SetIntraOpNumThreads(m->opts, 1));
        ORT_OK(m->api->SetInterOpNumThreads(m->opts, 1));
        ORT_OK(m->api->DisableCpuMemArena(m->opts));
        ORT_OK(m->api->AddSessionConfigEntry(m->opts, "session.intra_op.allow_spinning", "0"));
        ORT_OK(m->api->AddSessionConfigEntry(m->opts, "session.inter_op.allow_spinning", "0"));
        ORT_OK(m->api->SetSessionGraphOptimizationLevel(m->opts, ORT_ENABLE_ALL));
        m->npu = false;
        if (try_npu) {
            using NnapiFn = OrtStatus* (*)(OrtSessionOptions*, uint32_t);
            auto append_nnapi = reinterpret_cast<NnapiFn>(
                dlsym(m->lib, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
            // fp32 on purpose: relaxed fp16 drifts the recurrent state and is heard as
            // hiss. CPU_DISABLED keeps NNAPI off its reference CPU emulation; whatever
            // the accelerator cannot take (DFT, ...) runs on the ORT CPU EP.
            if (append_nnapi) {
                OrtStatus* nst = append_nnapi(m->opts, NNAPI_FLAG_CPU_DISABLED);
                if (!nst) {
                    m->npu = true;
                } else {
                    VLOGI("NNAPI EP unavailable (%s), using CPU", m->api->GetErrorMessage(nst));
                    m->api->ReleaseStatus(nst);
                }
            }
        }
        if (!ORT_OK(m->api->CreateSession(m->env, model_path, m->opts, &m->session))) {
            m->session = nullptr;
            m->npu = false;
            return false;
        }
        return true;
    };

    const bool try_npu = use_npu && has_nnapi_accelerator();
    if (use_npu && !try_npu) {
        VLOGI("no NNAPI accelerator registered, running the separator on CPU");
    }
    if (!(try_npu && create_session(true)) && !create_session(false)) {
        delete m;
        return nullptr;
    }
    // The session keeps what it needs; the options are no longer used.
    m->api->ReleaseSessionOptions(m->opts);
    m->opts = nullptr;

    ORT_OK(m->api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &m->mem_info));

    OrtAllocator* alloc = nullptr;
    ORT_OK(m->api->GetAllocatorWithDefaultOptions(&alloc));
    size_t num_in = 0, num_out = 0;
    ORT_OK(m->api->SessionGetInputCount(m->session, &num_in));
    ORT_OK(m->api->SessionGetOutputCount(m->session, &num_out));
    for (size_t i = 0; i < num_in; ++i) {
        char* name = nullptr;
        ORT_OK(m->api->SessionGetInputName(m->session, i, alloc, &name));
        m->input_names.emplace_back(name ? name : "");
        if (name) ORT_OK(m->api->AllocatorFree(alloc, name));
        if (i == 0) continue;
        OrtTypeInfo* ti = nullptr;
        const OrtTensorTypeAndShapeInfo* si = nullptr;
        size_t dims = 0;
        ORT_OK(m->api->SessionGetInputTypeInfo(m->session, i, &ti));
        ORT_OK(m->api->CastTypeInfoToTensorInfo(ti, &si));
        ORT_OK(m->api->GetDimensionsCount(si, &dims));
        std::vector<int64_t> shape(dims);
        ORT_OK(m->api->GetDimensions(si, shape.data(), dims));
        size_t total = 1;
        for (auto& dim : shape) {
            if (dim <= 0) dim = 1;
            total *= static_cast<size_t>(dim);
        }
        m->state_shapes.push_back(std::move(shape));
        m->state_sizes.push_back(total);
        m->api->ReleaseTypeInfo(ti);
    }
    for (size_t i = 0; i < num_out; ++i) {
        char* name = nullptr;
        ORT_OK(m->api->SessionGetOutputName(m->session, i, alloc, &name));
        m->output_names.emplace_back(name ? name : "");
        if (name) ORT_OK(m->api->AllocatorFree(alloc, name));
    }
    if (num_in < 1 || num_out != num_in) {
        VLOGE("unexpected model signature: %zu inputs, %zu outputs", num_in, num_out);
        delete m;
        return nullptr;
    }
    VLOGI("StemgenRT session ready (npu=%d, %zu state tensors)", m->npu ? 1 : 0, m->state_sizes.size());
    return m;
}

SharedModel* acquire_model(const char* model_path, bool use_npu) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (g_model == nullptr) {
        g_model = load_model(model_path, use_npu);
        if (g_model == nullptr) return nullptr;
    }
    g_model->refs++;
    return g_model;
}

void release_model(SharedModel* m) {
    if (m == nullptr) return;
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (--m->refs == 0) {
        if (g_model == m) g_model = nullptr;
        delete m;
    }
}

} // namespace

struct PipelinedVocals::Impl {
    SharedModel* m = nullptr;

    // Per-instance recurrent state; only the worker thread touches it.
    std::vector<std::vector<float>> state_bufs;
    HopBuf prev_in{};
    std::vector<const char*> in_names;
    std::vector<const char*> out_names;
    std::vector<OrtValue*> in_vals;
    std::vector<OrtValue*> out_vals;

    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
    std::thread worker;

    std::deque<QueuedHop> inbox;
    std::deque<OutHop> outbox;
    uint64_t gen = 0;
    uint64_t seq = 0;
    uint64_t last_tick = UINT64_MAX;

    // Audio-thread only.
    float wet = 0.0f;
    int cooldown = 0;
    std::vector<float> fallback_buf; // (1 + kMaxDelayHops) * 2 * kHop
    size_t fallback_pos = 0;

    std::atomic<uint32_t> stat_hops{0};
    std::atomic<uint32_t> stat_misses{0};
    std::atomic<uint32_t> stat_resets{0};

    void reset_states() {
        for (auto& b : state_bufs) std::fill(b.begin(), b.end(), 0.0f);
        prev_in.fill(0.0f);
    }

    bool run_model(const float* in_audio, float* out_dry, float* out_rest) {
        const OrtApi* api = m->api;
        const size_t n_states = state_bufs.size();
        in_vals.assign(n_states + 1, nullptr);
        out_vals.assign(n_states + 1, nullptr);
        auto release_all = [&]() {
            for (auto*& v : in_vals) { if (v) api->ReleaseValue(v); v = nullptr; }
            for (auto*& v : out_vals) { if (v) api->ReleaseValue(v); v = nullptr; }
        };

        static const int64_t kAudioShape[3] = {1, 2, kHop};
        OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
            m->mem_info, const_cast<float*>(in_audio), 2 * kHop * sizeof(float),
            kAudioShape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_vals[0]);
        for (size_t i = 0; !st && i < n_states; ++i) {
            st = api->CreateTensorWithDataAsOrtValue(
                m->mem_info, state_bufs[i].data(), state_bufs[i].size() * sizeof(float),
                m->state_shapes[i].data(), m->state_shapes[i].size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_vals[i + 1]);
        }
        if (!st) {
            st = api->Run(m->session, nullptr, in_names.data(), in_vals.data(), in_vals.size(),
                          out_names.data(), out_names.size(), out_vals.data());
        }
        if (st) {
            api->ReleaseStatus(st);
            release_all();
            return false;
        }

        float* vocals_data = nullptr;
        if (OrtStatus* gs = api->GetTensorMutableData(out_vals[0], reinterpret_cast<void**>(&vocals_data))) {
            api->ReleaseStatus(gs);
            release_all();
            return false;
        }
        std::memcpy(out_dry, prev_in.data(), 2 * kHop * sizeof(float));
        std::memcpy(prev_in.data(), in_audio, 2 * kHop * sizeof(float));
        bool finite = true;
        for (int i = 0; i < 2 * kHop; ++i) {
            const float v = vocals_data[i];
            finite = finite && std::isfinite(v);
            out_rest[i] = out_dry[i] - v;
        }
        for (size_t i = 0; i < n_states; ++i) {
            float* sdata = nullptr;
            if (OrtStatus* gs = api->GetTensorMutableData(out_vals[i + 1], reinterpret_cast<void**>(&sdata))) {
                api->ReleaseStatus(gs);
                finite = false;
                break;
            }
            std::memcpy(state_bufs[i].data(), sdata, state_bufs[i].size() * sizeof(float));
        }
        release_all();
        if (!finite) {
            // A NaN/Inf in the recurrent state would poison every following hop.
            reset_states();
            return false;
        }
        return true;
    }
};

PipelinedVocals::~PipelinedVocals() {
    if (loader_.joinable()) {
        loader_.join();
    }
    if (!impl_) return;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->cv.notify_all();
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
    release_model(impl_->m);
    delete impl_;
    impl_ = nullptr;
}

bool PipelinedVocals::ensure_loaded(const char* model_path, bool use_npu) {
    if (ready_.load()) return true;
    if (failed_.load()) return false;
    // Loading ONNX Runtime and compiling the graph takes hundreds of milliseconds;
    // doing it on the audio thread stalled playback the moment neural mode was
    // switched on. Load in the background and keep the passive path meanwhile.
    if (!loading_.exchange(true)) {
        model_path_ = model_path;
        loader_ = std::thread([this, use_npu]() { load(use_npu); });
    }
    return false;
}

void PipelinedVocals::load(bool use_npu) {
    SharedModel* m = acquire_model(model_path_.c_str(), use_npu);
    if (m == nullptr) {
        failed_.store(true);
        return;
    }
    auto* d = new Impl();
    d->m = m;
    for (size_t sz : m->state_sizes) d->state_bufs.emplace_back(sz, 0.0f);
    for (const auto& n : m->input_names) d->in_names.push_back(n.c_str());
    for (const auto& n : m->output_names) d->out_names.push_back(n.c_str());
    d->fallback_buf.assign((1 + kMaxDelayHops) * 2 * kHop, 0.0f);

    // Warm up (first Run allocates kernels and caches).
    HopBuf zeros{}, d_tmp{}, r_tmp{};
    for (int i = 0; i < 4; ++i) {
        d->run_model(zeros.data(), d_tmp.data(), r_tmp.data());
    }
    d->reset_states();

    npu_active_.store(m->npu);
    impl_ = d;
    d->worker = std::thread([this]() { worker_loop(); });
    ready_.store(true); // publishes impl_ to the audio thread
}

void PipelinedVocals::note_block_frames(int frames) {
    const int needed = std::clamp((frames + kHop - 1) / kHop + kMarginHops, kMinDelayHops, kMaxDelayHops);
    // Only grow: AudioFlinger occasionally hands over a short tail, which must not
    // make the pipeline shrink and resync.
    if (needed > delay_hops_) {
        delay_hops_ = needed;
        delay_changed_ = true;
        VLOGI("pipeline delay -> %d hops (block of %d frames)", delay_hops_, frames);
    }
}

void PipelinedVocals::worker_loop() {
    Impl* d = impl_; // set before this thread is started
    uint64_t worker_gen = UINT64_MAX;
    uint32_t runs = 0;
    double run_ms_sum = 0.0, run_ms_max = 0.0;
    auto last_log = std::chrono::steady_clock::now();
    while (true) {
        QueuedHop item;
        {
            std::unique_lock<std::mutex> lock(d->mutex);
            d->cv.wait(lock, [&]() { return d->stop || !d->inbox.empty(); });
            if (d->stop) break;
            item = d->inbox.front();
            d->inbox.pop_front();
        }

        if (item.gen != worker_gen) {
            d->reset_states();
            worker_gen = item.gen;
        }

        OutHop out;
        out.gen = item.gen;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = d->run_model(item.audio.data(), out.dry.data(), out.rest.data());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        runs++;
        run_ms_sum += ms;
        run_ms_max = std::max(run_ms_max, ms);
        if (ok && item.seq >= 1) {
            out.seq = item.seq - 1; // the model output lags its input by one hop
            std::lock_guard<std::mutex> lock(d->mutex);
            if (item.gen == d->gen) {
                d->outbox.push_back(out);
            }
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(t1 - last_log).count() >= 5) {
            VLOGI("separator: runs=%u avg=%.2fms max=%.2fms hops=%u misses=%u resets=%u npu=%d",
                  runs, runs ? run_ms_sum / runs : 0.0, run_ms_max,
                  d->stat_hops.exchange(0), d->stat_misses.exchange(0), d->stat_resets.exchange(0),
                  npu_active_.load() ? 1 : 0);
            runs = 0;
            run_ms_sum = run_ms_max = 0.0;
            last_log = t1;
        }
    }
}

void PipelinedVocals::passive_matrix(const float* dry, float* rest) {
    for (int i = 0; i < kHop; ++i) {
        float mid = 0.5f * (dry[i] + dry[kHop + i]);
        float v = 0.5f * mid;
        rest[i] = dry[i] - v;
        rest[kHop + i] = dry[kHop + i] - v;
    }
}

void PipelinedVocals::process(const float* stereo_in, float* dry_out, float* rest_out, uint64_t tick) {
    if (!ready_.load() || !impl_) {
        std::memcpy(dry_out, stereo_in, 2 * kHop * sizeof(float));
        passive_matrix(dry_out, rest_out);
        return;
    }
    Impl* d = impl_;
    d->stat_hops.fetch_add(1, std::memory_order_relaxed);
    const size_t delay_hops = static_cast<size_t>(delay_hops_);

    uint64_t cur_gen, cur_seq;
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        bool resync = delay_changed_ ||
                      (d->last_tick != UINT64_MAX && tick != d->last_tick + 1);
        // The worker fell far behind (device too slow): drop the backlog and restart
        // the model rather than letting the queue grow without bound.
        if (d->inbox.size() > 2 * delay_hops) {
            resync = true;
            d->stat_resets.fetch_add(1, std::memory_order_relaxed);
        }
        if (resync) {
            d->gen++;
            d->seq = 0;
            d->inbox.clear();
            d->outbox.clear();
            std::fill(d->fallback_buf.begin(), d->fallback_buf.end(), 0.0f);
            d->fallback_pos = 0;
            d->wet = 0.0f;
            delay_changed_ = false;
        }
        d->last_tick = tick;
        cur_seq = d->seq++;
        cur_gen = d->gen;
        QueuedHop& q = d->inbox.emplace_back();
        q.gen = cur_gen;
        q.seq = cur_seq;
        std::memcpy(q.audio.data(), stereo_in, 2 * kHop * sizeof(float));
    }
    d->cv.notify_one();

    // Fallback delay line: the passive path is delayed exactly like the model path,
    // so switching between them never jumps in time.
    const size_t ring = 1 + delay_hops;
    float* slot = &d->fallback_buf[d->fallback_pos * 2 * kHop];
    HopBuf dry_ref;
    std::memcpy(dry_ref.data(), slot, 2 * kHop * sizeof(float));
    std::memcpy(slot, stereo_in, 2 * kHop * sizeof(float));
    d->fallback_pos = (d->fallback_pos + 1) % ring;

    if (cur_seq < ring) {
        std::memcpy(dry_out, dry_ref.data(), 2 * kHop * sizeof(float));
        passive_matrix(dry_out, rest_out);
        return;
    }

    const uint64_t target_seq = cur_seq - ring;
    bool got = false;
    {
        std::lock_guard<std::mutex> lock(d->mutex);
        while (!d->outbox.empty() &&
               (d->outbox.front().gen != cur_gen || d->outbox.front().seq < target_seq)) {
            d->outbox.pop_front();
        }
        if (!d->outbox.empty() && d->outbox.front().seq == target_seq) {
            const OutHop& res = d->outbox.front();
            std::memcpy(dry_out, res.dry.data(), 2 * kHop * sizeof(float));
            std::memcpy(rest_out, res.rest.data(), 2 * kHop * sizeof(float));
            d->outbox.pop_front();
            got = true;
        }
    }

    if (!got) {
        d->stat_misses.fetch_add(1, std::memory_order_relaxed);
        d->cooldown = kMissCooldownHops;
    } else if (d->cooldown > 0) {
        d->cooldown--;
        got = false;
    }

    if (!got) {
        d->wet = 0.0f;
        std::memcpy(dry_out, dry_ref.data(), 2 * kHop * sizeof(float));
        passive_matrix(dry_out, rest_out);
        return;
    }

    if (d->wet < 1.0f) {
        // Fade the model in over ~10 ms (four hops) from the passive matrix.
        HopBuf p_rest;
        passive_matrix(dry_out, p_rest.data());
        const float end = std::min(1.0f, d->wet + 0.25f);
        const float step = (end - d->wet) / static_cast<float>(kHop);
        for (int i = 0; i < kHop; ++i) {
            const float w = d->wet + step * static_cast<float>(i + 1);
            rest_out[i] = p_rest[i] * (1.0f - w) + rest_out[i] * w;
            rest_out[kHop + i] = p_rest[kHop + i] * (1.0f - w) + rest_out[kHop + i] * w;
        }
        d->wet = end;
    }
}

} // namespace ally
