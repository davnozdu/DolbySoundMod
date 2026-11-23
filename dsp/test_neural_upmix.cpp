#include "spatial.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// Feed the renderer a deterministic, ready separator. This tests the routing
// through the normal stereo entry point without loading ORT on the host.
namespace ally {
PipelinedVocals::~PipelinedVocals() = default;
bool PipelinedVocals::ensure_loaded(const char*, bool) { return true; }
void PipelinedVocals::note_hop(const float*) {}
void PipelinedVocals::note_block_frames(int) {}
void PipelinedVocals::process(const float* input, float* dry, float* rest, uint64_t) {
    for (int i = 0; i < kHop; ++i) {
        const float voice = 0.5f * (input[i] + input[kHop + i]);
        dry[i] = input[i];
        dry[kHop + i] = input[kHop + i];
        rest[i] = input[i] - voice;
        rest[kHop + i] = input[kHop + i] - voice;
    }
}
} // namespace ally

std::vector<float> render(const std::vector<float>& input, bool neural, bool upmix) {
    ally::Renderer renderer;
    renderer.init(48000.0f);
    renderer.set_profile(ally::Profile::Movie);
    renderer.set_device(ally::DeviceKind::Headphones);
    renderer.set_reverb(false);
    renderer.set_neural(neural);
    renderer.set_upmix(upmix);
    std::vector<float> output(input.size());
    renderer.process_interleaved(input.data(), output.data(), input.size() / 2, 2);
    assert(renderer.upmix_active() == upmix);
    return output;
}

int main() {
    constexpr int frames = 48000 / 5;
    std::vector<float> input(2 * frames);
    for (int i = 0; i < frames; ++i) {
        const float voice = 0.16f * std::sin(i * 0.042f);
        const float ambience = 0.08f * std::sin(i * 0.081f);
        input[2 * i] = voice + ambience;
        input[2 * i + 1] = voice - ambience;
    }

    const auto both = render(input, true, true);
    const auto neural_only = render(input, true, false);
    const auto upmix_only = render(input, false, true);
    double energy = 0.0, upmix_difference = 0.0, neural_difference = 0.0;
    for (size_t i = 2048; i < both.size(); ++i) {
        assert(std::isfinite(both[i]) && std::abs(both[i]) <= 1.01f);
        energy += both[i] * both[i];
        upmix_difference += std::pow(both[i] - neural_only[i], 2);
        neural_difference += std::pow(both[i] - upmix_only[i], 2);
    }
    assert(energy > 0.1 && upmix_difference > 0.1 && neural_difference > 0.1);
    std::puts("Neural movie and exact upmix both affect the stereo render.");
}
