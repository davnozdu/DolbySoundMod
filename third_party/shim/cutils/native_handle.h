#pragma once
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef struct native_handle {
    int version;
    int numFds;
    int numInts;
    int data[0];
} native_handle_t;
static inline native_handle_t* native_handle_create(int nfd, int nint_) {
    if (nfd < 0 || nint_ < 0 || nfd > 4096 || nint_ > 4096) return nullptr;
    native_handle_t* h = (native_handle_t*)calloc(1, sizeof(native_handle_t) + sizeof(int) * (nfd + nint_));
    if (!h) return nullptr;
    h->version = sizeof(native_handle_t);
    h->numFds = nfd;
    h->numInts = nint_;
    return h;
}
static inline int native_handle_delete(native_handle_t* h) { if (!h) return -1; free(h); return 0; }
static inline int native_handle_close(const native_handle_t* h) {
    for (int i = 0; i < h->numFds; ++i) close(h->data[i]);
    return 0;
}
