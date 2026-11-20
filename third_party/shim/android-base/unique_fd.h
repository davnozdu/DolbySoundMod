#pragma once
#include <unistd.h>
namespace android { namespace base {
class unique_fd {
  public:
    unique_fd() : fd_(-1) {}
    explicit unique_fd(int fd) : fd_(fd) {}
    unique_fd(const unique_fd&) = delete;
    unique_fd& operator=(const unique_fd&) = delete;
    unique_fd(unique_fd&& o) : fd_(o.release()) {}
    unique_fd& operator=(unique_fd&& o) { reset(o.release()); return *this; }
    ~unique_fd() { reset(); }
    int get() const { return fd_; }
    int release() { int f = fd_; fd_ = -1; return f; }
    void reset(int fd = -1) { if (fd_ >= 0 && fd_ != fd) close(fd_); fd_ = fd; }
    explicit operator bool() const { return fd_ >= 0; }
    bool operator==(int fd) const { return fd_ == fd; }
    bool operator!=(int fd) const { return fd_ != fd; }
    bool operator<(int fd) const { return fd_ < fd; }
  private:
    int fd_;
};
}}
