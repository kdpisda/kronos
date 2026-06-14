#pragma once
// ============================================================================
// KRONOS  src/gpu/gpu_context.hpp
// GPU context singleton: device selection, handle management.
//
// Manages:
//   - Device selection (local_rank-based for MPI+GPU)
//   - cuBLAS/rocBLAS handle lifecycle
//   - Deterministic mode configuration
// ============================================================================

#include <string>

namespace kronos::gpu {

class GPUContext {
public:
    /// Get the singleton instance.
    static GPUContext& instance();

    /// Initialize the GPU context.
    /// mpi_rank: global MPI rank (for logging)
    /// local_rank: node-local rank (for GPU device selection)
    void init(int mpi_rank = 0, int local_rank = 0);

    /// Finalize and release GPU resources.
    void finalize();

    /// Check if the context has been initialized.
    bool is_initialized() const { return initialized_; }

    /// Get the selected GPU device ID.
    int device_id() const { return device_id_; }

    /// Get the cuBLAS/rocBLAS handle (as opaque pointer).
    void* blas_handle() const { return blas_handle_; }

    /// Get the Metal command queue (opaque MTL::CommandQueue*). Nullptr on non-Metal builds.
    void* metal_queue() const { return metal_queue_; }

    /// Get the number of available GPU devices.
    int num_devices() const { return num_devices_; }

    /// Get a description string of the selected GPU.
    const std::string& device_name() const { return device_name_; }

    /// Enable deterministic mode (cuBLAS workspace config).
    void set_deterministic(bool enable);

private:
    GPUContext() = default;
    ~GPUContext();

    // Non-copyable
    GPUContext(const GPUContext&) = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    bool initialized_{false};
    int device_id_{0};
    int num_devices_{0};
    void* blas_handle_{nullptr};
    void* metal_queue_{nullptr};  // MTL::CommandQueue* on Metal builds
    std::string device_name_{"none"};
};

} // namespace kronos::gpu
