#pragma once

#include <memory>

namespace ally {

class AutoEq {
public:
    AutoEq();
    ~AutoEq();
    void init(float rate);
    void reset();
    void set_enabled(bool enabled);
    // Music profile: refine music by genre. Off (films, games, podcasts): speech vs music only.
    void set_genre(bool enabled);
    bool enabled() const;
    void observe(const float* left, const float* right);
    // No audio callback for `frames` frames (paused/stopped): releases the models.
    void idle(int frames);
    void apply(float* left, float* right);
    const char* profile() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    float rate_ = 48000.0f;
    bool genre_ = false;
};

} // namespace ally
