#include "spatial.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    ally::Renderer r;
    r.init(48000.0f);

    constexpr int kFrames = 4800; // 100 ms
    std::vector<float> in(kFrames * 2, 0.0f);
    std::vector<float> out(kFrames * 2, 0.0f);
    for (int i = 0; i < kFrames; ++i) {
        float t = static_cast<float>(i) / 48000.0f;
        in[2 * i + 0] = 0.3f * std::sin(2.0f * 3.14159265f * 440.0f * t);
        in[2 * i + 1] = 0.25f * std::sin(2.0f * 3.14159265f * 550.0f * t);
    }

    for (auto prof : {ally::Profile::Game, ally::Profile::Movie, ally::Profile::Music,
                      ally::Profile::Speech}) {
        for (auto dev : {ally::DeviceKind::Headphones, ally::DeviceKind::Speakers, ally::DeviceKind::Glasses}) {
            r.set_profile(prof);
            r.set_device(dev);
            r.process_interleaved(in.data(), out.data(), kFrames, 2);
            double energy = 0.0;
            for (float v : out) {
                assert(std::isfinite(v));
                assert(std::abs(v) <= 1.01f);
                energy += v * v;
            }
            assert(energy > 0.1);
        }
    }
    std::printf("DSP test passed: all profiles & devices finite and bounded.\n");
    return 0;
}
