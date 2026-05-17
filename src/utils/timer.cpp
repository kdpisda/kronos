#include "utils/timer.hpp"
#include "utils/mpi_wrapper.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace kronos {

// ---------------------------------------------------------------------------
// TimerRegistry
// ---------------------------------------------------------------------------

TimerRegistry& TimerRegistry::instance() {
    static TimerRegistry registry;
    return registry;
}

void TimerRegistry::record(const std::string& name, double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = entries_[name];
    if (entry.name.empty()) {
        entry.name = name;
    }
    entry.total_seconds += seconds;
    entry.call_count += 1;
}

const std::map<std::string, TimingEntry>& TimerRegistry::entries() const {
    return entries_;
}

void TimerRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}

void TimerRegistry::print_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (entries_.empty()) {
        std::cout << "  (no timings recorded)\n";
        return;
    }

    // Collect entries and sort by total time descending
    std::vector<const TimingEntry*> sorted;
    sorted.reserve(entries_.size());
    for (const auto& [key, entry] : entries_) {
        sorted.push_back(&entry);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const TimingEntry* a, const TimingEntry* b) {
                  return a->total_seconds > b->total_seconds;
              });

    // Find the longest name for formatting
    size_t max_name_len = 4; // minimum "Name"
    for (const auto* e : sorted) {
        max_name_len = std::max(max_name_len, e->name.size());
    }

    // Print header
    std::cout << std::left << std::setw(static_cast<int>(max_name_len + 2)) << "Name"
              << std::right << std::setw(14) << "Total (s)"
              << std::setw(10) << "Calls"
              << std::setw(14) << "Avg (s)"
              << "\n";
    std::cout << std::string(max_name_len + 2 + 14 + 10 + 14, '-') << "\n";

    // Print rows
    for (const auto* e : sorted) {
        double avg = (e->call_count > 0) ? e->total_seconds / e->call_count : 0.0;
        std::cout << std::left << std::setw(static_cast<int>(max_name_len + 2)) << e->name
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(14) << e->total_seconds
                  << std::setw(10) << e->call_count
                  << std::setw(14) << avg
                  << "\n";
    }
}

void TimerRegistry::print_summary_mpi() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (entries_.empty()) {
        if (mpi::rank() == 0) {
            std::cout << "  (no timings recorded)\n";
        }
        return;
    }

    // Collect entries sorted by total time descending (local)
    std::vector<const TimingEntry*> sorted;
    sorted.reserve(entries_.size());
    for (const auto& [key, entry] : entries_) {
        sorted.push_back(&entry);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const TimingEntry* a, const TimingEntry* b) {
                  return a->total_seconds > b->total_seconds;
              });

    // MPI reduce min/max/avg for each timer
    int nprocs = mpi::size();
    for (const auto* e : sorted) {
        double t_min = e->total_seconds;
        double t_max = e->total_seconds;
        double t_sum = e->total_seconds;

        mpi::allreduce_min_inplace(&t_min, 1);
        mpi::allreduce_max_inplace(&t_max, 1);
        mpi::allreduce_sum_inplace(&t_sum, 1);

        if (mpi::rank() == 0) {
            if (e == sorted.front()) {
                // Print header only once
                std::cout << std::left << std::setw(24) << "Timer"
                          << std::right << std::setw(12) << "Min (s)"
                          << std::setw(12) << "Max (s)"
                          << std::setw(12) << "Avg (s)"
                          << std::setw(8) << "Calls"
                          << "\n";
                std::cout << std::string(68, '-') << "\n";
            }
            double avg = t_sum / nprocs;
            std::cout << std::left << std::setw(24) << e->name
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << t_min
                      << std::setw(12) << t_max
                      << std::setw(12) << avg
                      << std::setw(8) << e->call_count
                      << "\n";
        }
    }
}

std::map<std::string, double> TimerRegistry::as_map() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, double> result;
    for (const auto& [key, entry] : entries_) {
        result[key] = entry.total_seconds;
    }
    return result;
}

// ---------------------------------------------------------------------------
// ScopedTimer
// ---------------------------------------------------------------------------

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start_;
    TimerRegistry::instance().record(name_, elapsed.count());
}

} // namespace kronos
