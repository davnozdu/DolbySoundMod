#pragma once
// libfmq asks for an ashmem region; this kernel has no ashmem ioctls at all
// ("ashmem set name: Inappropriate ioctl for device"), so provide the same
// anonymous shared memory through memfd instead. The mapping semantics the FMQ
// cares about (MAP_SHARED over an anonymous file, fds passed to another process)
// are identical, and the module's sepolicy already allows the audio HAL domain to
// create, size and map its own files.

#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>

static inline int ashmem_create_region(const char* name, size_t size) {
    (void)name;
    int fd = static_cast<int>(
            ::syscall(__NR_memfd_create, "fmq", MFD_CLOEXEC | MFD_ALLOW_SEALING));
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
