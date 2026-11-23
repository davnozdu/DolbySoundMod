#pragma once

#include <onnxruntime_c_api.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unistd.h>

namespace ally {

// Serialize access to HTP within the HAL. Inference workers can use their CPU
// sessions if another model is preparing or running a graph.
inline std::mutex& qnn_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Open>
bool with_qnn_library_path(const std::string& dir, Open&& open) {
    const std::lock_guard<std::mutex> lock(qnn_mutex());
    const char* env = std::getenv("ADSP_LIBRARY_PATH");
    const bool had_path = env != nullptr;
    const std::string previous = env ? env : "";
    const std::string path = had_path ? dir + ";" + previous : dir;
    if (setenv("ADSP_LIBRARY_PATH", path.c_str(), 1) != 0) return false;
    const bool ready = open();
    if (had_path) setenv("ADSP_LIBRARY_PATH", previous.c_str(), 1);
    else unsetenv("ADSP_LIBRARY_PATH");
    return ready;
}

// QNN HTP performance votes. Battery first: the background classifier asks for the
// lowest clocks; the real-time mask decoder (40 ms budget, ~2 ms of work) asks for a
// balanced vote instead of burst.
constexpr const char* kQnnBackgroundMode = "power_saver";
constexpr const char* kQnnRealtimeMode = "balanced";

// Compiled HTP graphs, written by the HAL itself (post-fs-data.sh creates the directory
// as audio:audio and empties it at boot, so a module update never meets a stale graph).
constexpr const char* kQnnCacheDir = "/data/vendor/audio/allysound_qnn/";

// Creates a QNN session from the cached EP-context model when there is one. Otherwise
// it prepares the graph (seconds of CPU in libQnnHtpPrepare) and, when the cache
// directory is writable, saves the result so the next load - e.g. after the separator
// was unloaded for silence - skips preparation entirely. `qnn_options` must already
// carry the QNN execution provider.
inline bool create_cached_qnn_session(const OrtApi* api, OrtEnv* env, OrtSessionOptions* qnn_options,
                                      const std::string& model, const std::string& cache_name,
                                      OrtSession** session,
                                      const std::function<bool(OrtStatus*)>& ok) {
    const std::string cache = std::string(kQnnCacheDir) + cache_name;
    if (access(cache.c_str(), R_OK) == 0) {
        if (ok(api->CreateSession(env, cache.c_str(), qnn_options, session))) return true;
        unlink(cache.c_str());  // stale or damaged: prepare again below
        *session = nullptr;
    }
    const bool writable = access(kQnnCacheDir, W_OK) == 0;
    const std::string temporary = cache + ".tmp";
    OrtSessionOptions* options = qnn_options;
    OrtSessionOptions* dump = nullptr;
    if (writable && ok(api->CloneSessionOptions(qnn_options, &dump))) {
        unlink(temporary.c_str());
        if (ok(api->AddSessionConfigEntry(dump, "ep.context_enable", "1")) &&
            ok(api->AddSessionConfigEntry(dump, "ep.context_embed_mode", "1")) &&
            ok(api->AddSessionConfigEntry(dump, "ep.context_file_path", temporary.c_str()))) {
            options = dump;
        }
    }
    bool created = ok(api->CreateSession(env, model.c_str(), options, session));
    const bool dumped = created && options != qnn_options;
    if (!created && options != qnn_options) {
        // Context dumping is an optimization; never let it cost the session.
        *session = nullptr;
        created = ok(api->CreateSession(env, model.c_str(), qnn_options, session));
    }
    if (dump) api->ReleaseSessionOptions(dump);
    if (dumped && access(temporary.c_str(), R_OK) == 0) {
        std::rename(temporary.c_str(), cache.c_str());
    } else {
        unlink(temporary.c_str());
    }
    return created;
}

} // namespace ally
