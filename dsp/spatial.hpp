#pragma once

#include "fft.hpp"
#ifdef ALLY_ENABLE_NPU
#include "vocals.hpp"
#endif
#ifdef ALLY_ENABLE_AUTO_EQ
#include "auto_eq.hpp"
#endif
#include <array>
#include <complex>
#include <cstdint>
#include <vector>

namespace ally {

constexpr int kHop = 128;
constexpr int kNumSpeakers = 7; // FL, FR, FC, RL, RR, SL, SR

enum class Profile : int {
    Off = 0,
    Game = 1,
    Movie = 2,
    Music = 3,
    Speech = 4,  // podcast / audiobook: dry, forward, compression instead of space
};

enum class DeviceKind : int {
    Headphones = 0,
    Speakers = 1,
    Glasses = 2,
};

// Direct-Form II Transposed biquad / IIR filter for multi-channel HOP blocks.
class BiquadFilter {
public:
    BiquadFilter() = default;
    void configure_lowpass_2nd(float cutoff_hz, float sample_rate, int channels);
    void configure_highpass_2nd(float cutoff_hz, float sample_rate, int channels);
    void configure_highpass_1st(float cutoff_hz, float sample_rate, int channels);
    void configure_lowpass_1st(float cutoff_hz, float sample_rate, int channels);
    void configure_bandpass_crosstalk(float sample_rate);
    void configure_high_shelf(float freq_hz, float gain_db, float sample_rate, int channels);
    void reset();
    void process(const float* in, float* out, int ch, int frames);
    void process_inplace(float* buf, int ch, int frames);

private:
    // Generic IIR up to order 4 (for cascaded 2nd+2nd + FIR taps in crosstalk)
    std::vector<float> b_;
    std::vector<float> a_;
    std::vector<float> state_; // channels * max_order
    int channels_ = 0;
    int order_ = 0;

    friend class CrosstalkWidener;
};

class DelayLine {
public:
    DelayLine() = default;
    void init(int channels, int delay_samples);
    void reset();
    void process(const float* in, float* out, int frames);

private:
    int channels_ = 0;
    int delay_ = 0;
    int pos_ = 0;
    std::vector<float> buffer_;
};

class Convolver {
public:
    Convolver() = default;
    // responses: [sources * ears * length]
    void init(int sources, int ears, int length, const float* responses, bool diagonal = false);
    void set_responses(int length, const float* responses);
    void reset();
    // in: [sources * kHop], out: [ears * kHop] (added or overwritten)
    void process(const float* in, float* out);

private:
    int sources_ = 0;
    int ears_ = 0;
    int parts_ = 0;
    bool diagonal_ = false;
    RealFFT fft_{2 * kHop};
    std::vector<std::complex<float>> spectra_;  // [parts, sources, ears, bins]
    std::vector<std::complex<float>> history_;  // [parts, sources, bins]
    std::vector<float> previous_;               // [sources, kHop]
    std::vector<std::complex<float>> accum_;    // [ears, bins]
    std::vector<float> frame_buf_;              // [2 * kHop]
};

class Dynamics {
public:
    void init(float sample_rate);
    void reset();
    // stereo: 2 * kHop planar
    void process(float* left, float* right, bool leveler, bool compress);

private:
    float sample_rate_ = 48000.0f;
    BiquadFilter k_high_;
    BiquadFilter k_shelf_;
    DelayLine lookahead_;
    float mean_square_ = 0.01f;
    float momentary_ = 0.0f;
    float grow_ = 0.0f;
    float fall_ = 0.0f;
    float relax_ = 0.0f;
    float level_gain_db_ = 0.0f;
    float envelope_db_ = -90.0f;
    float attack_ = 0.0f;
    float release_ = 0.0f;
    float pending_ = 1.0f;
    float gain_ = 1.0f;

    std::array<float, kHop> w_left_{};
    std::array<float, kHop> w_right_{};
    std::array<float, kHop> s_left_{};
    std::array<float, kHop> s_right_{};
    std::array<float, 2 * kHop> del_in_{};
    std::array<float, 2 * kHop> del_out_{};

    void update_leveler(const float* left, const float* right, bool enabled);
    float compute_compressor(const float* left, const float* right, float gain_db);
};

// Coherence-based stereo upmix (primary/ambient decomposition in the STFT domain).
//
// Per frequency band the inter-channel coherence of the stereo pair splits it into a
// correlated (direct) part and an uncorrelated (ambient) part; the two masks sum to
// one, so direct + ambient reproduces the input exactly. The direct part is placed by
// its level panning: centred content goes to the centre speaker, hard-panned content
// stays on its own front speaker (the passive matrix leaks it, polarity-inverted, into
// the opposite surrounds). The ambient part feeds the sides and rears.
// STFT: 512-point sqrt-Hann, hop kHop (75% overlap), latency 384 samples (8 ms).
class CoherenceUpmix {
public:
    static constexpr int kN = 4 * kHop;
    static constexpr int kBins = kN / 2 + 1;

    void init(float sample_rate);
    void reset();
    // l, r: one hop of input; outputs one hop each (delayed by kN - kHop samples).
    void process(const float* l, const float* r, float* front_l, float* front_r,
                 float* centre, float* amb_l, float* amb_r);

private:
    static constexpr int kOut = 5;  // front L/R, centre, ambient L/R
    RealFFT fft_{kN};
    float alpha_ = 0.9f;
    std::vector<float> window_;                         // sqrt periodic Hann
    std::array<std::vector<float>, 2> history_;         // kN per channel
    std::array<std::vector<std::complex<float>>, 2> spec_;
    std::array<std::vector<std::complex<float>>, kOut> out_spec_;
    std::array<std::vector<float>, kOut> ola_;          // kN per output
    std::vector<float> frame_;
    std::vector<float> pll_, prr_;
    std::vector<std::complex<float>> plr_;
};

class CrosstalkWidener {
public:
    void init(float sample_rate);
    void reset();
    void process(float* left, float* right);

private:
    BiquadFilter filter_;
    std::array<float, kHop> side_{};
    std::array<float, kHop> extra_{};
};

class Renderer {
public:
    void init(float sample_rate);
    void reset();

    void set_profile(Profile p);
    void set_device(DeviceKind d);
    void set_neural(bool enabled) { neural_ = enabled; }
    void set_reverb(bool enabled) { reverb_ = enabled; }
    void set_bypass(bool bypass) { bypass_ = bypass; }
    // Coherence upmix for stereo sources, including the bed of a neural mix.
    void set_upmix(bool enabled);
    bool upmix() const { return upmix_on_; }
    bool upmix_active() const {
        return upmix_on_ && !is_surround_ && !bypass_ && profile_ != Profile::Off;
    }

    Profile profile() const { return profile_; }
    DeviceKind device() const { return device_; }
    bool neural() const { return neural_; }
    bool reverb() const { return reverb_; }
#ifdef ALLY_ENABLE_AUTO_EQ
    void set_auto_eq(bool enabled) { auto_eq_.set_enabled(enabled); }
    const char* auto_eq_profile() const { return auto_eq_.profile(); }
#endif
    bool npu_active() const {
#ifdef ALLY_ENABLE_NPU
        return vocals_.using_npu();
#else
        return false;
#endif
    }

    // Process arbitrary number of interleaved float frames (2, 6, or 8 channels).
    // Output is always interleaved stereo float frames.
    void process_interleaved(const float* in, float* out, int frames, int in_channels);

    // The host had no audio for `frames` frames: lets the separator unload when idle.
    void idle(int frames) {
        (void)frames;
#ifdef ALLY_ENABLE_NPU
        vocals_.idle(frames);
#endif
#ifdef ALLY_ENABLE_AUTO_EQ
        auto_eq_.idle(frames);
#endif
    }

private:
    float sample_rate_ = 48000.0f;
    Profile profile_ = Profile::Game;
    DeviceKind device_ = DeviceKind::Headphones;
    bool reverb_ = true;
    bool bypass_ = false;
    bool upmix_on_ = false;
    bool neural_ = false;
    uint64_t tick_ = 0;

#ifdef ALLY_ENABLE_NPU
    PipelinedVocals vocals_;
    DelayLine upmix_dialogue_delay_;
    bool upmix_uses_neural_ = false;
#endif
#ifdef ALLY_ENABLE_AUTO_EQ
    AutoEq auto_eq_;
#endif
    BiquadFilter cinema_air_;
    BiquadFilter speech_hp_;
    BiquadFilter speech_presence_;

    Convolver direct_;
    Convolver room_;
    float room_rt60_ = 0.0f; // rt60 of the response currently loaded into room_

    DelayLine ambience_delay_;
    DelayLine speaker_delay_;
    BiquadFilter lfe_filter_;
    BiquadFilter speaker_lfe_;
    BiquadFilter glasses_bass_;
    CrosstalkWidener widener_;
    CoherenceUpmix upmix_;
    Dynamics dynamics_;

    double clock_ = 0.0;
    double surround_seen_ = -1e9;
    bool is_surround_ = false;

    // Residual FIFO so arbitrary Android frame counts (e.g. 192, 240, 256, 480)
    // are processed in exact kHop (128) blocks with zero artifacts.
    int fifo_fill_ = 0;
    std::array<std::array<float, kHop>, 8> in_fifo_{};
    std::array<std::array<float, kHop>, 2> out_fifo_{};
    int out_read_pos_ = 0;
    int out_avail_ = 0;

    // Scratch buffers for one kHop block
    std::array<float, 8 * kHop> block_{};
    std::array<float, 7 * kHop> feeds_{};
    std::array<float, kHop> lfe_{};
    std::array<float, 2 * kHop> ears_{};
    std::array<float, 2 * kHop> room_send_{};
    std::array<float, 2 * kHop> room_out_{};

    void process_hop();
    bool detect_surround();
    void make_stereo_feeds(bool& has_lfe);
    void make_surround_feeds(bool& has_lfe);
    void render_headphones(bool has_lfe, float* out_l, float* out_r);
    void render_speakers(bool has_lfe, float* out_l, float* out_r);
    void apply_speech_eq(float* out_l, float* out_r);
    void downmix(float* out_l, float* out_r);
};

} // namespace ally
