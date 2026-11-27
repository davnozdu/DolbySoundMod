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

    for (bool scene : {false, true}) {
    r.set_scene_adapter(scene);
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
    }

    // Changing the output route during steady audio should not step the waveform.
    {
        constexpr int kBefore = 256 * ally::kHop;
        constexpr int kAfter = 8 * ally::kHop;
        constexpr int kBoundary = kBefore + 2 * ally::kHop - 1;
        std::vector<float> input(2 * (kBefore + kAfter)), output(input.size());
        for (int phase = 0; phase < 8; ++phase) {
            ally::Renderer renderer;
            renderer.init(48000.0f);
            renderer.set_profile(ally::Profile::Music);
            renderer.set_reverb(false);
            for (int i = 0; i < kBefore + kAfter; ++i) {
                input[2 * i] = 0.12f * std::sin(0.0061f * i + 0.7854f * phase);
                input[2 * i + 1] = 0.10f * std::sin(0.0082f * i + 0.7854f * phase);
            }
            renderer.process_interleaved(input.data(), output.data(), kBefore, 2);
            renderer.set_device(ally::DeviceKind::Speakers);
            renderer.process_interleaved(input.data() + 2 * kBefore, output.data() + 2 * kBefore, kAfter, 2);
            for (int channel = 0; channel < 2; ++channel) {
                const int sample = 2 * kBoundary + channel;
                assert(std::abs(output[sample] - output[sample - 2]) < 0.002f);
            }
        }
    }

    // Scene adapter: verify 2.0 -> 7.1 music placement produces side & rear feeds.
    {
        ally::CoherenceUpmix u;
        u.init(48000.0f);
        constexpr int H = ally::kHop, kHops = 80;
        float fl[H], fr[H], c[H], sl[H], sr[H], rl[H], rr[H];
        double side_e = 0.0;
        for (int h = 0; h < kHops; ++h) {
            float L[H], R[H];
            for (int i = 0; i < H; ++i) {
                const float t = static_cast<float>(h * H + i) / 48000.0f;
                const float kick = 0.25f * std::sin(2.0f * 3.14159265f * 65.0f * t);
                const float guitar_l = 0.25f * std::sin(2.0f * 3.14159265f * 1200.0f * t);
                const float guitar_r = 0.25f * std::sin(2.0f * 3.14159265f * 1800.0f * t);
                L[i] = kick + guitar_l;
                R[i] = kick + guitar_r;
            }
            u.process_scene(L, R, ally::Profile::Music, fl, fr, c, sl, sr, rl, rr);
            if (h > 8) {
                for (int i = 0; i < H; ++i) {
                    side_e += sl[i] * sl[i] + sr[i] * sr[i];
                }
            }
        }
        assert(side_e > 1.0);
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
