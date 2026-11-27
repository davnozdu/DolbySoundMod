#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

namespace ally {

// Self-contained radix-2 real-to-complex FFT for power-of-two sizes (N = 256).
// Produces N/2 + 1 complex bins matching numpy.fft.rfft / irfft scaling.
class RealFFT {
public:
    explicit RealFFT(size_t n) : n_(n), half_(n / 2), rfft_twiddles_(half_) {
        constexpr double kPi = 3.14159265358979323846;
        for (size_t k = 0; k < half_; ++k) {
            double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(n_);
            rfft_twiddles_[k] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
        }
        scratch_.resize(half_);
        // Each transform uses the same permutation of the packed complex samples.
        for (size_t i = 1, j = 0; i < half_; ++i) {
            size_t bit = half_ >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swaps_.emplace_back(i, j);
        }
        // Precompute half-size complex FFT twiddles
        half_twiddles_.resize(half_ / 2);
        for (size_t k = 0; k < half_ / 2; ++k) {
            double angle = -2.0 * kPi * static_cast<double>(k) / static_cast<double>(half_);
            half_twiddles_[k] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
        }
    }

    size_t size() const { return n_; }
    size_t bins() const { return half_ + 1; }

    // Forward real FFT: in[n] -> out[n/2 + 1]
    void forward(const float* in, std::complex<float>* out) const {
        auto& buf = scratch_; // preallocated: no heap traffic on the audio thread
        for (size_t i = 0; i < half_; ++i) {
            buf[i] = {in[2 * i], in[2 * i + 1]};
        }
        fft_complex(buf.data(), half_, false);

        out[0] = {buf[0].real() + buf[0].imag(), 0.0f};
        out[half_] = {buf[0].real() - buf[0].imag(), 0.0f};

        for (size_t k = 1; k < half_; ++k) {
            const auto& zk = buf[k];
            const auto zmk = std::conj(buf[half_ - k]);
            const auto even = 0.5f * (zk + zmk);
            const auto odd_j = 0.5f * (zk - zmk); // = j * odd
            const auto odd = std::complex<float>(odd_j.imag(), -odd_j.real());
            out[k] = even + rfft_twiddles_[k] * odd;
        }
    }

    // Inverse real FFT: in[n/2 + 1] -> out[n] (scaled by 1/n, matching numpy.fft.irfft)
    void inverse(const std::complex<float>* in, float* out) const {
        auto& buf = scratch_; // preallocated: no heap traffic on the audio thread
        const float x0 = in[0].real();
        const float xn = in[half_].real();
        buf[0] = {0.5f * (x0 + xn), 0.5f * (x0 - xn)};

        for (size_t k = 1; k < half_; ++k) {
            const auto& xk = in[k];
            const auto xmk = std::conj(in[half_ - k]);
            const auto even = 0.5f * (xk + xmk);
            const auto odd_tw = 0.5f * (xk - xmk);
            const auto odd = std::conj(rfft_twiddles_[k]) * odd_tw;
            const auto odd_j = std::complex<float>(-odd.imag(), odd.real());
            buf[k] = even + odd_j;
        }

        fft_complex(buf.data(), half_, true);
        const float inv = 1.0f / static_cast<float>(half_);
        for (size_t i = 0; i < half_; ++i) {
            out[2 * i] = buf[i].real() * inv;
            out[2 * i + 1] = buf[i].imag() * inv;
        }
    }

private:
    void fft_complex(std::complex<float>* data, size_t n, bool inv) const {
        for (const auto& [i, j] : swaps_) std::swap(data[i], data[j]);
        // Cooley-Tukey butterfly
        for (size_t len = 2; len <= n; len <<= 1) {
            size_t half = len >> 1;
            size_t step = n / len;
            for (size_t i = 0; i < n; i += len) {
                for (size_t j = 0; j < half; ++j) {
                    auto w = half_twiddles_[j * step];
                    if (inv) w = std::conj(w);
                    auto u = data[i + j];
                    auto v = data[i + j + half] * w;
                    data[i + j] = u + v;
                    data[i + j + half] = u - v;
                }
            }
        }
    }

    size_t n_;
    size_t half_;
    std::vector<std::pair<size_t, size_t>> swaps_;
    std::vector<std::complex<float>> rfft_twiddles_;
    std::vector<std::complex<float>> half_twiddles_;
    // Scratch for forward()/inverse(). Each Convolver owns its RealFFT and calls it
    // from one thread at a time, so a shared mutable buffer is safe.
    mutable std::vector<std::complex<float>> scratch_;
};

} // namespace ally
