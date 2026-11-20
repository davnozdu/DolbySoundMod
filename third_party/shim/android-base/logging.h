#pragma once
// Minimal stand-in for libbase, used only by the vendored libfmq sources: the effect
// library must not depend on platform C++ libraries (different libc++ ABI namespace),
// so libbase is replaced by these few lines.

#include <android/log.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sstream>
#include <string>

static inline void android_errorWriteLog(int tag, const char* info) {
    (void)tag;
    (void)info;
}

static inline void android_errorWriteWithInfoLog(int tag, const char* subTag, int uid,
                                                 const char* data, int dataLen) {
    (void)tag;
    (void)subTag;
    (void)uid;
    (void)data;
    (void)dataLen;
}

namespace android {
namespace base {

enum class LogSeverity : int { VERBOSE = 2, DEBUG = 3, INFO = 4, WARNING = 5, ERROR = 6, FATAL = 7 };

class LogMessage {
  public:
    explicit LogMessage(LogSeverity sev) : sev_(sev) {}
    ~LogMessage() {
        const std::string text = stream_.str();
        if (sev_ == LogSeverity::FATAL) {
            __android_log_print(ANDROID_LOG_FATAL, "libfmq", "%s", text.c_str());
            abort();
        }
        __android_log_print(static_cast<int>(sev_), "libfmq", "%s", text.c_str());
    }
    std::ostream& stream() { return stream_; }

  private:
    LogSeverity sev_;
    std::ostringstream stream_;
};

}  // namespace base
}  // namespace android

#define LOG(severity) ::android::base::LogMessage(::android::base::LogSeverity::severity).stream()

// Stream-shaped like libbase's CHECK, because libfmq writes `CHECK(exp) << message;`.
#define CHECK(condition)                                                                    \
    if (condition) {                                                                        \
    } else                                                                                  \
        ::android::base::LogMessage(::android::base::LogSeverity::FATAL).stream()
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define PLOG(severity) LOG(severity)
