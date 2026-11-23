// Host test: the split separator (C++ FFT front/back end + ONNX Runtime backbone)
// must match the full StemgenRT model run by ONNX Runtime.
//
// usage: test_vocals_split FULL_DIR/vocals.onnx SPLIT_DIR/vocals.onnx
//   SPLIT_DIR also holds vocals_backbone.onnx and vocals_dsp.bin (tools/make_cpu_model.py)
#include "vocals.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using ally::PipelinedVocals;
constexpr int H = PipelinedVocals::kHop;
constexpr int kHops = 48000 * 3 / H;  // 3 s
constexpr double kPi = 3.14159265358979323846;

// Deterministic stereo test signal: a gliding harmonic "voice" in the centre over
// panned noise and a bass line, so both the vocal and the rest paths carry energy.
static void make_hop(int h, float* out) {
    for (int i = 0; i < H; ++i) {
        const double t = (h * H + i) / 48000.0;
        const double f0 = 180.0 + 40.0 * std::sin(2.0 * kPi * 0.7 * t);
        double voice = 0.0;
        for (int k = 1; k <= 6; ++k) voice += std::sin(2.0 * kPi * f0 * k * t) / k;
        // Hash of the sample index: both runs must see exactly the same signal.
        uint32_t rng = static_cast<uint32_t>(h * H + i) * 2654435761u;
        rng ^= rng >> 15;
        rng *= 2246822519u;
        rng ^= rng >> 13;
        const double noise = ((rng >> 8) / 16777216.0 - 0.5) * 0.1;
        const double bass = 0.2 * std::sin(2.0 * kPi * 55.0 * t);
        out[i] = static_cast<float>(0.15 * voice + bass + noise);
        out[H + i] = static_cast<float>(0.15 * voice + bass - 0.7 * noise);
    }
}

static bool run(const char* model, std::vector<float>& rest_all) {
    PipelinedVocals v;
    std::vector<float> in(2 * H), dry(2 * H), rest(2 * H);
    make_hop(0, in.data());
    v.note_hop(in.data());
    for (int w = 0; w < 5000 && !v.ensure_loaded(model, false); ++w) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!v.is_ready()) {
        std::fprintf(stderr, "failed to load %s\n", model);
        return false;
    }
    v.note_block_frames(H);
    for (int h = 0; h < kHops; ++h) {
        make_hop(h, in.data());
        v.note_hop(in.data());
        v.process(in.data(), dry.data(), rest.data(), static_cast<uint64_t>(h) + 1);
        rest_all.insert(rest_all.end(), rest.begin(), rest.end());
        // Feed slowly so the worker never misses: the outputs then depend on the
        // model alone, not on scheduling.
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s FULL/vocals.onnx SPLIT/vocals.onnx\n", argv[0]);
        return 2;
    }
    std::vector<float> full, split;
    if (!run(argv[1], full) || !run(argv[2], split)) return 1;
    // Skip the first second: the model warms up from zero state there.
    double e = 0.0, s = 0.0;
    for (size_t i = 2 * H * 375; i < full.size(); ++i) {
        const double d = static_cast<double>(full[i]) - split[i];
        e += d * d;
        s += static_cast<double>(full[i]) * full[i];
    }
    const double db = 10.0 * std::log10(e / (s + 1e-30) + 1e-30);
    std::printf("split vs full separator: %.1f dB\n", db);
    if (!(db < -80.0)) {
        std::fprintf(stderr, "split separator does not match the full model\n");
        return 1;
    }
    std::printf("split separator test passed\n");
    return 0;
}
