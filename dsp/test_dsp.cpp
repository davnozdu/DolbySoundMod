#include "spatial.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
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

    for (bool upmix : {false, true}) {
    r.set_upmix(upmix);
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
    }

    // Coherence upmix: front + centre + ambient must rebuild the input exactly.
    {
        ally::CoherenceUpmix u;
        u.init(48000.0f);
        constexpr int H = ally::kHop, kHops = 400, D = ally::CoherenceUpmix::kN - H;
        std::vector<float> L(kHops * H), R(kHops * H), oL(kHops * H), oR(kHops * H);
        uint32_t seed = 1;
        for (int i = 0; i < kHops * H; ++i) {
            seed = seed * 1664525u + 1013904223u;
            const float n = (seed >> 8) / 16777216.0f - 0.5f;
            const float t = static_cast<float>(i) / 48000.0f;
            L[i] = 0.3f * std::sin(2.0f * 3.14159265f * 300.0f * t) + 0.2f * n;
            R[i] = 0.1f * std::sin(2.0f * 3.14159265f * 300.0f * t) - 0.2f * n;
        }
        float fl[H], fr[H], c[H], al[H], ar[H];
        for (int h = 0; h < kHops; ++h) {
            u.process(&L[h * H], &R[h * H], fl, fr, c, al, ar);
            for (int i = 0; i < H; ++i) {
                oL[h * H + i] = fl[i] + c[i] + al[i];
                oR[h * H + i] = fr[i] + c[i] + ar[i];
            }
        }
        double e = 0.0, s = 0.0;
        for (int i = 8 * H + D; i < kHops * H; ++i) {
            e += std::pow(oL[i] - L[i - D], 2) + std::pow(oR[i] - R[i - D], 2);
            s += L[i - D] * L[i - D] + R[i - D] * R[i - D];
        }
        const double db = 10.0 * std::log10(e / s);
        std::printf("upmix reconstruction: %.1f dB\n", db);
        assert(db < -100.0);
    }
    std::printf("DSP test passed: all profiles & devices finite and bounded, with and without upmix.\n");
    return 0;
}
