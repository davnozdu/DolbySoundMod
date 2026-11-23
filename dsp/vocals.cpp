#include "vocals.hpp"
#include "fft.hpp"

#include <dlfcn.h>
#include <unistd.h>
#include <onnxruntime_c_api.h>
#include <nnapi_provider_factory.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
constexpr int kQnnHops = 15;
constexpr int kQnnMarginHops = 10;
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

// Share sessions across effects using the same model and acceleration setting.
// Each effect owns its own recurrent state.
struct SharedModel {
    void* lib = nullptr;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* opts = nullptr;
    OrtSession* session = nullptr;
    OrtSession* qnn_core = nullptr;
    OrtSession* qnn_decoder = nullptr;
    OrtSession* cpu_decoder = nullptr;
    std::atomic<bool> qnn_healthy{false};
    OrtMemoryInfo* mem_info = nullptr;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> state_shapes;
    std::vector<size_t> state_sizes;
    std::vector<int64_t> in0_shape;
    size_t in0_size = 0;
    bool npu = false;
    int refs = 0;

    // Split model (tools/make_cpu_model.py): ONNX Runtime only runs the backbone;
    // the FFT front end and the mask/overlap-add back end run here, sharing one
    // real FFT per hop instead of three generic DFTs inside the graph.
    bool split = false;
    std::vector<float> analysis;     // kFft
    std::vector<float> synthesis;    // 2 * kHop
    std::vector<float> denominator;  // kHop
    float eps_add = 0.0f;
    float clip_min = 0.0f;

    ~SharedModel() {
        if (api) {
            if (cpu_decoder) api->ReleaseSession(cpu_decoder);
            if (qnn_decoder) api->ReleaseSession(qnn_decoder);
            if (qnn_core) api->ReleaseSession(qnn_core);
            if (session) api->ReleaseSession(session);
            if (opts) api->ReleaseSessionOptions(opts);
            if (mem_info) api->ReleaseMemoryInfo(mem_info);
            if (env) api->ReleaseEnv(env);
        }
        if (lib) dlclose(lib);
    }
};

std::mutex g_model_mutex;
std::map<std::pair<std::string, bool>, SharedModel*> g_models;

constexpr int kFft = 1024;
constexpr int kBins = kFft / 2 + 1;
constexpr int kHistory = kFft - kHop;

bool load_dsp_params(const std::string& path, SharedModel* m) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char magic[8] = {};
    uint32_t version = 0;
    m->analysis.resize(kFft);
    m->synthesis.resize(2 * kHop);
    m->denominator.resize(kHop);
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "ALLYVDSP", 8) == 0 &&
              std::fread(&version, 4, 1, f) == 1 && version == 1 &&
              std::fread(m->analysis.data(), 4, kFft, f) == static_cast<size_t>(kFft) &&
              std::fread(m->synthesis.data(), 4, 2 * kHop, f) == static_cast<size_t>(2 * kHop) &&
              std::fread(m->denominator.data(), 4, kHop, f) == static_cast<size_t>(kHop) &&
              std::fread(&m->eps_add, 4, 1, f) == 1 && std::fread(&m->clip_min, 4, 1, f) == 1;
    std::fclose(f);
    if (!ok) VLOGE("bad DSP parameter file %s", path.c_str());
    return ok;
}

#define ORT_OK(expr) (([&]() { OrtStatus* _s = (expr); if (_s) { VLOGE("%s: %s", #expr, m->api->GetErrorMessage(_s)); m->api->ReleaseStatus(_s); return false; } return true; })())

bool load_qnn(SharedModel* m, const std::string& dir) {
    const std::string backend = dir + "libQnnHtp.so";
    if (!ORT_OK(m->api->AddSessionConfigEntry(m->opts, "session.disable_cpu_ep_fallback", "1"))) return false;
    const char* previous_path = std::getenv("ADSP_LIBRARY_PATH");
    const bool had_path = previous_path != nullptr;
    const std::string dsp_path = dir + (previous_path ? ";" + std::string(previous_path) : "");
    if (setenv("ADSP_LIBRARY_PATH", dsp_path.c_str(), 1) != 0) return false;
    const std::string core = dir + "vocals_core.onnx";
    const std::string decoder = dir + "vocals_decoder.onnx";
    const char* keys[] = {"backend_path", "enable_htp_fp16_precision", "htp_performance_mode"};
    const char* values[] = {backend.c_str(), "1", "burst"};
    const bool ready = ORT_OK(m->api->SessionOptionsAppendExecutionProvider(m->opts, "QNN", keys, values, 3)) &&
                       ORT_OK(m->api->CreateSession(m->env, decoder.c_str(), m->opts, &m->qnn_decoder));
    if (had_path) setenv("ADSP_LIBRARY_PATH", dsp_path.c_str() + dir.size() + 1, 1);
    else unsetenv("ADSP_LIBRARY_PATH");
    if (!ready) return false;

    OrtSessionOptions* cpu = nullptr;
    if (!ORT_OK(m->api->CreateSessionOptions(&cpu))) return false;
    const bool cpu_ready = ORT_OK(m->api->SetIntraOpNumThreads(cpu, 1)) &&
                           ORT_OK(m->api->DisableCpuMemArena(cpu)) &&
                           ORT_OK(m->api->AddSessionConfigEntry(cpu, "session.intra_op.allow_spinning", "0")) &&
                           ORT_OK(m->api->CreateSession(m->env, core.c_str(), cpu, &m->qnn_core)) &&
                           ORT_OK(m->api->CreateSession(m->env, decoder.c_str(), cpu, &m->cpu_decoder));
    m->api->ReleaseSessionOptions(cpu);
    m->qnn_healthy.store(cpu_ready);
    return cpu_ready;
}

SharedModel* load_model(const char* full_model_path, bool use_npu) {
    auto* m = new SharedModel();
    std::string path = full_model_path;
    {
        const std::string dir = path.substr(0, path.find_last_of('/') + 1);
        const std::string backbone = dir + "vocals_backbone.onnx";
        if (path.find("vocals.onnx") != std::string::npos && access(backbone.c_str(), R_OK) == 0 &&
            load_dsp_params(dir + "vocals_dsp.bin", m)) {
            path = backbone;
            m->split = true;
        }
    }
    const char* model_path = path.c_str();
    const std::string qnn_dir = path.substr(0, path.find_last_of('/') + 1) + "ally-qnn/";
    const std::string qnn_runtime = qnn_dir + "libonnxruntime.so";
    const bool qnn_requested = use_npu && m->split && access(qnn_runtime.c_str(), R_OK) == 0;
    if (qnn_requested) {
        m->lib = dlopen(qnn_runtime.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!m->lib) VLOGE("QNN runtime load failed: %s", dlerror());
    }
    const bool qnn_runtime_loaded = m->lib != nullptr;
    static const char* kOrtCandidates[] = {
        "libonnxruntime.so",
        "/vendor/lib64/soundfx/libonnxruntime.so",
        "/vendor/lib64/libonnxruntime.so",
    };
    for (const char* cand : kOrtCandidates) {
        if (m->lib) break;
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

    const bool try_npu = use_npu && !qnn_requested && has_nnapi_accelerator();
    if (use_npu && !qnn_requested && !try_npu) {
        VLOGI("no NNAPI accelerator registered, running the separator on CPU");
    }
    if (!(try_npu && create_session(true)) && !create_session(false)) {
        delete m;
        return nullptr;
    }
    if (qnn_runtime_loaded) {
        if (load_qnn(m, qnn_dir)) {
            VLOGI("QNN decoder ready: FP32 CPU recurrent core, HTP mask decoder, %d hops", kQnnHops);
        } else {
            VLOGE("QNN hybrid initialization failed; using the CPU separator");
        }
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
        if (i == 0) {
            m->in0_shape = std::move(shape);
            m->in0_size = total;
        } else {
            m->state_shapes.push_back(std::move(shape));
            m->state_sizes.push_back(total);
        }
        m->api->ReleaseTypeInfo(ti);
    }
    for (size_t i = 0; i < num_out; ++i) {
        char* name = nullptr;
        ORT_OK(m->api->SessionGetOutputName(m->session, i, alloc, &name));
        m->output_names.emplace_back(name ? name : "");
        if (name) ORT_OK(m->api->AllocatorFree(alloc, name));
    }
    const size_t want_in0 = m->split ? 2 * kBins * 3 : 2 * kHop;
    if (num_in < 1 || num_out != num_in || m->in0_size != want_in0) {
        VLOGE("unexpected model signature: %zu inputs, %zu outputs, input0 %zu", num_in, num_out, m->in0_size);
        delete m;
        return nullptr;
    }
    VLOGI("StemgenRT session ready (%s, %s, %zu state tensors)", m->split ? "split" : "full",
          m->qnn_healthy.load() ? "QNN hybrid" : m->npu ? "NNAPI" : "CPU", m->state_sizes.size());
    return m;
}

SharedModel* acquire_model(const char* model_path, bool use_npu) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    const std::pair<std::string, bool> key{model_path, use_npu};
    auto it = g_models.find(key);
    if (it == g_models.end()) {
        SharedModel* model = load_model(model_path, use_npu);
        if (!model) return nullptr;
        it = g_models.emplace(key, model).first;
    }
    it->second->refs++;
    return it->second;
}

void release_model(SharedModel* m) {
    if (m == nullptr) return;
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (--m->refs == 0) {
        std::erase_if(g_models, [m](const auto& entry) { return entry.second == m; });
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

    // Split-model front/back end (worker thread only).
    ally::RealFFT fft{kFft};
    std::vector<float> history;                  // 2 * kHistory
    std::vector<float> tail;                     // 2 * kHop
    std::vector<float> frame;                    // kFft
    std::vector<float> features;                 // [2][kBins][3]
    std::vector<std::complex<float>> spectrum;   // [2][kBins]
    std::vector<std::complex<float>> masked;     // kBins
    std::vector<float> synth;                    // kFft
    HopBuf vocals{};
    std::vector<float> qnn_features;
    std::vector<std::complex<float>> qnn_spectra;
    bool batched = false;

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
        std::fill(history.begin(), history.end(), 0.0f);
        std::fill(tail.begin(), tail.end(), 0.0f);
    }

    // Front end: windowed real FFT of [history | chunk] per channel, then the
    // model's features [Re/s, Im/s, log(1 + |X|/s)] with s the RMS spectrum level.
    void front_end(const float* in_audio) {
        double power = 0.0;
        for (int c = 0; c < 2; ++c) {
            float* h = &history[c * kHistory];
            for (int i = 0; i < kHistory; ++i) frame[i] = h[i] * m->analysis[i];
            for (int i = 0; i < kHop; ++i) frame[kHistory + i] = in_audio[c * kHop + i] * m->analysis[kHistory + i];
            std::complex<float>* X = &spectrum[c * kBins];
            fft.forward(frame.data(), X);
            for (int k = 0; k < kBins; ++k) power += std::norm(X[k]);
            std::memmove(h, h + kHop, (kHistory - kHop) * sizeof(float));
            std::memcpy(h + kHistory - kHop, &in_audio[c * kHop], kHop * sizeof(float));
        }
        const float mean = static_cast<float>(power / (2.0 * kBins));
        const float inv = 1.0f / std::sqrt(std::max(mean, m->clip_min));
        for (int c = 0; c < 2; ++c) {
            const std::complex<float>* X = &spectrum[c * kBins];
            float* f = &features[c * kBins * 3];
            for (int k = 0; k < kBins; ++k) {
                const float re = X[k].real(), im = X[k].imag();
                f[3 * k + 0] = re * inv;
                f[3 * k + 1] = im * inv;
                f[3 * k + 2] = std::log1p(std::sqrt(re * re + im * im + m->eps_add) * inv);
            }
        }
    }

    // Back end: apply the complex mask, inverse FFT, synthesis window, overlap-add.
    void back_end(const float* mask) {
        for (int c = 0; c < 2; ++c) {
            const std::complex<float>* X = &spectrum[c * kBins];
            const float* mk = &mask[c * kBins * 2];
            for (int k = 0; k < kBins; ++k) {
                masked[k] = X[k] * std::complex<float>(mk[2 * k], mk[2 * k + 1]);
            }
            fft.inverse(masked.data(), synth.data());
            float* t = &tail[c * kHop];
            float* v = &vocals[c * kHop];
            const float* seg = &synth[kFft - 2 * kHop];
            for (int i = 0; i < kHop; ++i) {
                v[i] = (seg[i] * m->synthesis[i] + t[i]) / m->denominator[i];
                t[i] = seg[kHop + i] * m->synthesis[kHop + i];
            }
        }
    }

    bool run_qnn_batch(const std::array<QueuedHop, kQnnHops>& items,
                       std::array<OutHop, kQnnHops>& outputs) {
        const OrtApi* api = m->api;
        constexpr size_t feature_size = 2 * kBins * 3;
        constexpr size_t mask_size = 2 * kBins * 2;
        constexpr int64_t features_shape[] = {kQnnHops, 1, 1, 2, kBins, 3};
        constexpr int64_t local_shape[] = {2, 20, 96};
        constexpr int64_t global_shape[] = {2, 1, 192};
        constexpr int64_t decode_shape[] = {kQnnHops, 20, 96};
        const char* core_inputs[] = {"features", "vocals_local_hidden", "vocals_global_hidden"};
        const char* core_outputs[] = {"encoded", "next_vocals_local_hidden", "next_vocals_global_hidden"};
        const char* decoder_input = "encoded";
        const char* decoder_output = "mask";
        for (size_t i = 0; i < items.size(); ++i) {
            front_end(items[i].audio.data());
            std::copy(features.begin(), features.end(), qnn_features.begin() + i * feature_size);
            std::copy(spectrum.begin(), spectrum.end(), qnn_spectra.begin() + i * 2 * kBins);
        }

        OrtValue* inputs[3]{};
        OrtValue* core[3]{};
        OrtValue* encoded = nullptr;
        OrtValue* mask = nullptr;
        auto tensor = [&](std::vector<float>& values, const auto& shape, OrtValue** out) {
            return ORT_OK(api->CreateTensorWithDataAsOrtValue(
                m->mem_info, values.data(), values.size() * sizeof(float),
                shape, std::size(shape), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out));
        };
        auto data = [&](OrtValue* value, size_t size, float** samples) {
            OrtTensorTypeAndShapeInfo* shape = nullptr;
            if (!ORT_OK(api->GetTensorTypeAndShape(value, &shape))) return false;
            size_t count = 0;
            ONNXTensorElementDataType type;
            const bool matches = ORT_OK(api->GetTensorShapeElementCount(shape, &count)) &&
                                 ORT_OK(api->GetTensorElementType(shape, &type)) &&
                                 type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && count == size;
            api->ReleaseTensorTypeAndShapeInfo(shape);
            return matches && ORT_OK(api->GetTensorMutableData(value, reinterpret_cast<void**>(samples))) &&
                   std::all_of(*samples, *samples + size, [](float x) { return std::isfinite(x); });
        };
        const bool have_core =
            tensor(qnn_features, features_shape, &inputs[0]) &&
            tensor(state_bufs[0], local_shape, &inputs[1]) &&
            tensor(state_bufs[1], global_shape, &inputs[2]) &&
            ORT_OK(api->Run(m->qnn_core, nullptr, core_inputs, inputs, 3, core_outputs, 3, core));
        float* encoding = nullptr;
        float* local = nullptr;
        float* global = nullptr;
        float* masks = nullptr;
        constexpr size_t encoded_size = kQnnHops * 20 * 96;
        bool ok = have_core && data(core[0], encoded_size, &encoding) &&
                  data(core[1], state_bufs[0].size(), &local) &&
                  data(core[2], state_bufs[1].size(), &global) &&
                  ORT_OK(api->CreateTensorWithDataAsOrtValue(
                      m->mem_info, encoding, encoded_size * sizeof(float), decode_shape,
                      std::size(decode_shape), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &encoded));
        if (ok) {
            auto decode = [&](OrtSession* session) {
                return ORT_OK(api->Run(session, nullptr, &decoder_input, &encoded, 1, &decoder_output, 1, &mask)) &&
                       data(mask, kQnnHops * mask_size, &masks);
            };
            ok = m->qnn_healthy.load() && decode(m->qnn_decoder);
            if (!ok) {
                if (m->qnn_healthy.exchange(false)) VLOGE("QNN decoder failed; switching to CPU mask decoder");
                if (mask) { api->ReleaseValue(mask); mask = nullptr; }
                ok = decode(m->cpu_decoder);
            }
        }
        if (ok) {
            std::copy_n(local, state_bufs[0].size(), state_bufs[0].begin());
            std::copy_n(global, state_bufs[1].size(), state_bufs[1].begin());
            for (size_t i = 0; i < items.size(); ++i) {
                std::copy_n(qnn_spectra.begin() + i * 2 * kBins, 2 * kBins, spectrum.begin());
                back_end(masks + i * mask_size);
                outputs[i].dry = prev_in;
                prev_in = items[i].audio;
                for (size_t j = 0; j < prev_in.size(); ++j) {
                    outputs[i].rest[j] = outputs[i].dry[j] - vocals[j];
                    ok = ok && std::isfinite(outputs[i].rest[j]);
                }
            }
        }
        if (mask) api->ReleaseValue(mask);
        if (encoded) api->ReleaseValue(encoded);
        for (auto* v : core) if (v) api->ReleaseValue(v);
        for (auto* v : inputs) if (v) api->ReleaseValue(v);
        if (!ok) reset_states();
        return ok;
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

        const float* model_in = in_audio;
        if (m->split) {
            front_end(in_audio);
            model_in = features.data();
        }
        OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
            m->mem_info, const_cast<float*>(model_in), m->in0_size * sizeof(float),
            m->in0_shape.data(), m->in0_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_vals[0]);
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
        if (m->split) {
            back_end(vocals_data); // output 0 is the mask
            vocals_data = vocals.data();
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
    teardown();
}

void PipelinedVocals::teardown() {
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

void PipelinedVocals::unload() {
    // Only a finished load is torn down; one still in flight is left alone and
    // unloaded on a later idle hop.
    if (!ready_.load()) return;
    if (loader_.joinable()) loader_.join();
    ready_.store(false);
    teardown();
    npu_active_.store(false);
    loading_.store(false);
    VLOGI("separator unloaded after %d silent hops", silent_hops_);
}

void PipelinedVocals::note_hop(const float* stereo_in) {
    double e = 0.0;
    for (int i = 0; i < 2 * kHop; ++i) e += static_cast<double>(stereo_in[i]) * stereo_in[i];
    const bool silent = std::sqrt(e / (2 * kHop)) <= kSilenceRms;
    silent_hops_ = silent ? std::min(silent_hops_ + 1, kUnloadHops) : 0;
    if (silent_hops_ >= kUnloadHops) unload();
}

void PipelinedVocals::idle(int frames) {
    silent_hops_ = std::min(silent_hops_ + std::max(1, frames / kHop), kUnloadHops);
    if (silent_hops_ >= kUnloadHops) unload();
}

bool PipelinedVocals::ensure_loaded(const char* model_path, bool use_npu) {
    if (ready_.load()) return true;
    if (failed_.load()) return false;
    // Never load (or reload after an idle unload) for silence.
    if (silent_hops_ > 0) return false;
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
    if (m->split) {
        d->history.assign(2 * kHistory, 0.0f);
        d->tail.assign(2 * kHop, 0.0f);
        d->frame.assign(kFft, 0.0f);
        d->features.assign(2 * kBins * 3, 0.0f);
        d->spectrum.assign(2 * kBins, {0.0f, 0.0f});
        d->masked.assign(kBins, {0.0f, 0.0f});
        d->synth.assign(kFft, 0.0f);
    }
    for (const auto& n : m->input_names) d->in_names.push_back(n.c_str());
    for (const auto& n : m->output_names) d->out_names.push_back(n.c_str());
    d->fallback_buf.assign((1 + kMaxDelayHops) * 2 * kHop, 0.0f);

    // Warm up (first Run allocates kernels and caches).
    HopBuf zeros{}, d_tmp{}, r_tmp{};
    for (int i = 0; i < 4; ++i) {
        d->run_model(zeros.data(), d_tmp.data(), r_tmp.data());
    }
    d->reset_states();

    if (m->qnn_healthy.load() && m->state_sizes == std::vector<size_t>{2 * 20 * 96, 2 * 192}) {
        d->qnn_features.resize(kQnnHops * 2 * kBins * 3);
        d->qnn_spectra.resize(kQnnHops * 2 * kBins);
        const std::array<QueuedHop, kQnnHops> input{};
        std::array<OutHop, kQnnHops> output;
        d->batched = d->run_qnn_batch(input, output) && m->qnn_healthy.load();
        d->reset_states();
        if (!d->batched) VLOGE("QNN warm-up failed; using the CPU separator");
    }
    npu_active_.store(m->npu || d->batched);
    impl_ = d;
    d->worker = std::thread([this]() { worker_loop(); });
    ready_.store(true); // publishes impl_ to the audio thread
}

void PipelinedVocals::note_block_frames(int frames) {
    const int hops = std::max(1, (frames + kHop - 1) / kHop);
    int requested = hops + kMarginHops;
    if (ready_.load() && impl_->batched) {
        // A decoder batch can straddle callbacks when callback and batch widths differ.
        requested = std::max(hops, kQnnHops) + kQnnMarginHops +
                    (hops == kQnnHops ? 0 : std::min(hops, kQnnHops) - 1);
    }
    const int needed = std::clamp(requested, kMinDelayHops, kMaxDelayHops);
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
    const size_t count = d->batched ? kQnnHops : 1;
    std::array<QueuedHop, kQnnHops> items;
    std::array<OutHop, kQnnHops> outputs;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(d->mutex);
            d->cv.wait(lock, [&]() { return d->stop || d->inbox.size() >= count; });
            if (d->stop) break;
            for (size_t i = 0; i < count; ++i) {
                items[i] = d->inbox.front();
                d->inbox.pop_front();
            }
        }

        if (items[0].gen != worker_gen) {
            d->reset_states();
            worker_gen = items[0].gen;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = d->batched ? d->run_qnn_batch(items, outputs) :
            d->run_model(items[0].audio.data(), outputs[0].dry.data(), outputs[0].rest.data());
        if (d->batched) npu_active_.store(d->m->qnn_healthy.load());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        runs++;
        run_ms_sum += ms;
        run_ms_max = std::max(run_ms_max, ms);
        if (ok) {
            std::lock_guard<std::mutex> lock(d->mutex);
            if (worker_gen == d->gen) {
                for (size_t i = 0; i < count; ++i) {
                    if (items[i].seq == 0) continue;
                    outputs[i].gen = worker_gen;
                    outputs[i].seq = items[i].seq - 1; // inference has a one-hop lag
                    d->outbox.push_back(outputs[i]);
                }
            }
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(t1 - last_log).count() >= 5) {
            VLOGI("separator: runs=%u batch=%zu avg=%.2fms max=%.2fms hops=%u misses=%u resets=%u npu=%d",
                  runs, count, runs ? run_ms_sum / runs : 0.0, run_ms_max,
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
    // Silence: once the pipeline has drained there is nothing left to separate, so
    // the model is not run at all. The tick gap makes the pipeline resync and fade
    // the model back in when sound returns.
    if (silent_hops_ > delay_hops_ + 2) {
        std::memset(dry_out, 0, 2 * kHop * sizeof(float));
        std::memset(rest_out, 0, 2 * kHop * sizeof(float));
        return;
    }
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
