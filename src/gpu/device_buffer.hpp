#pragma once
// ============================================================================
// KRONOS  src/gpu/device_buffer.hpp
// RAII device buffer template for GPU memory management.
//
// DeviceBuffer<T> owns a GPU allocation and provides upload/download/move.
// In CPU-only builds, all operations throw GPUNotAvailableError.
// ============================================================================

#include "gpu/memory.hpp"
#include <cstddef>
#include <vector>
#include <utility>

namespace kronos::gpu {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    /// Allocate n elements on the GPU.
    explicit DeviceBuffer(size_t n) : size_(n) {
        if (n > 0) {
            ptr_ = static_cast<T*>(gpu_malloc(n * sizeof(T)));
        }
    }

    /// Move constructor
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    /// Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() { free(); }

    /// Upload from host vector to device.
    void upload(const std::vector<T>& host_data) {
        if (host_data.size() != size_) {
            free();
            size_ = host_data.size();
            if (size_ > 0) {
                ptr_ = static_cast<T*>(gpu_malloc(size_ * sizeof(T)));
            }
        }
        if (size_ > 0) {
            gpu_memcpy_h2d(ptr_, host_data.data(), size_ * sizeof(T));
        }
    }

    /// Upload from raw host pointer.
    void upload(const T* host_ptr, size_t count) {
        if (count != size_) {
            free();
            size_ = count;
            if (size_ > 0) {
                ptr_ = static_cast<T*>(gpu_malloc(size_ * sizeof(T)));
            }
        }
        if (size_ > 0) {
            gpu_memcpy_h2d(ptr_, host_ptr, size_ * sizeof(T));
        }
    }

    /// Download from device to host vector.
    std::vector<T> download() const {
        std::vector<T> host_data(size_);
        if (size_ > 0) {
            gpu_memcpy_d2h(host_data.data(), ptr_, size_ * sizeof(T));
        }
        return host_data;
    }

    /// Download to existing host buffer.
    void download(T* host_ptr) const {
        if (size_ > 0) {
            gpu_memcpy_d2h(host_ptr, ptr_, size_ * sizeof(T));
        }
    }

    /// Get raw device pointer.
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    void free() {
        if (ptr_) {
            gpu_free(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    T* ptr_{nullptr};
    size_t size_{0};
};

} // namespace kronos::gpu
