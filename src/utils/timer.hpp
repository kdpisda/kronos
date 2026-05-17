#pragma once
#include <string>
#include <chrono>
#include <map>
#include <vector>
#include <mutex>

namespace kronos {

// Accumulated timing statistics
struct TimingEntry {
    std::string name;
    double total_seconds{0.0};
    int call_count{0};
};

// Global timer registry (singleton)
class TimerRegistry {
public:
    static TimerRegistry& instance();

    void record(const std::string& name, double seconds);
    const std::map<std::string, TimingEntry>& entries() const;
    void reset();

    // Print summary to stdout
    void print_summary() const;

    // Print summary with MPI min/max/avg reduction (rank 0 prints)
    void print_summary_mpi() const;

    // Return as JSON-compatible structure
    std::map<std::string, double> as_map() const;

private:
    TimerRegistry() = default;
    std::map<std::string, TimingEntry> entries_;
    mutable std::mutex mutex_;
};

// RAII scoped timer - records elapsed time on destruction
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

    // Non-copyable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

// Convenience macro
#define KRONOS_TIMER(name) kronos::ScopedTimer _timer_##__LINE__(name)

} // namespace kronos
