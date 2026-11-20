#pragma once
// libfmq asks libcutils for an ashmem region. Two things matter for us:
//
//  1. The descriptor a vendor effect hands back is consumed by AudioFlinger, and this
//     device's (Qualcomm) conversion helper accepts the region only when it really is
//     ashmem: a memfd-backed queue is rejected and the effect is dropped with
//     "handleSetConfig ... Failed" right after open(). That is what the stock effect
//     library hands over, so we hand over the same kind of fd.
//  2. /dev/ashmem is not present on every kernel. Fall back to memfd, which has the same
//     mapping semantics (MAP_SHARED over an anonymous file, fds passed to another process).
//
// Note: the ashmem ioctls are issued only when the device node exists; anything else goes
// straight to the memfd path so a missing or restricted node cannot break effect creation.

#include <fcntl.h>
#include <linux/memfd.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>

#ifndef ASHMEM_NAME_LEN
#define ASHMEM_NAME_LEN 256
#endif
#ifndef ASHMEM_SET_NAME
#define ASHMEM_SET_NAME _IOW('a', 1, char[ASHMEM_NAME_LEN])
#endif
#ifndef ASHMEM_SET_SIZE
#define ASHMEM_SET_SIZE _IOW('a', 3, size_t)
#endif

static inline int ashmem_create_region(const char* name, size_t size) {
    int fd = ::open("/dev/ashmem", O_RDWR | O_CLOEXEC);
    if (fd >= 0) {
        char region[ASHMEM_NAME_LEN] = {};
        snprintf(region, sizeof(region), "%s", (name && *name) ? name : "MessageQueue");
        if (::ioctl(fd, ASHMEM_SET_NAME, region) == 0 &&
            ::ioctl(fd, ASHMEM_SET_SIZE, size) == 0) {
            return fd;
        }
        const int saved = errno;
        ::close(fd);
        errno = saved;
    }

    fd = static_cast<int>(::syscall(__NR_memfd_create, "fmq", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (fd < 0) return fd;
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        return -1;
    }
    return fd;
}

static inline int ashmem_set_prot_region(int fd, int prot) {
    (void)fd;
    (void)prot;
    return 0;
}
