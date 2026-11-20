#pragma once
#include <android/log.h>
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "libfmq", __VA_ARGS__)
#define ALOGW(...) __android_log_print(ANDROID_LOG_WARN, "libfmq", __VA_ARGS__)
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, "libfmq", __VA_ARGS__)
#define ALOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "libfmq", __VA_ARGS__)
#define ALOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "libfmq", __VA_ARGS__)
