// ============================================================================
// KRONOS  test/test_utils.cpp
// Tests for timer, logger, and utility classes.
// ============================================================================

#include <gtest/gtest.h>
#include "utils/timer.hpp"
#include "utils/logger.hpp"
#include "core/constants.hpp"

#include <chrono>
#include <sstream>
#include <thread>

using namespace kronos;

// ============================================================================
// TimerRegistry tests
// ============================================================================

TEST(Timer, RecordAndRetrieve) {
    auto& reg = TimerRegistry::instance();
    reg.reset();

    reg.record("test_op", 0.5);
    reg.record("test_op", 0.3);

    auto& entries = reg.entries();
    auto it = entries.find("test_op");
    ASSERT_NE(it, entries.end());
    EXPECT_NEAR(it->second.total_seconds, 0.8, 1e-10);
    EXPECT_EQ(it->second.call_count, 2);

    reg.reset();
}

TEST(Timer, ResetClearsAll) {
    auto& reg = TimerRegistry::instance();
    reg.record("temp_op", 1.0);
    EXPECT_FALSE(reg.entries().empty());

    reg.reset();
    EXPECT_TRUE(reg.entries().empty());
}

TEST(Timer, AsMapReturnsDoubles) {
    auto& reg = TimerRegistry::instance();
    reg.reset();

    reg.record("op_a", 1.5);
    reg.record("op_b", 2.5);

    auto map = reg.as_map();
    EXPECT_EQ(map.size(), 2u);
    EXPECT_NEAR(map["op_a"], 1.5, 1e-10);
    EXPECT_NEAR(map["op_b"], 2.5, 1e-10);

    reg.reset();
}

TEST(Timer, MultipleOperations) {
    auto& reg = TimerRegistry::instance();
    reg.reset();

    reg.record("alpha", 0.1);
    reg.record("beta", 0.2);
    reg.record("alpha", 0.15);

    auto& entries = reg.entries();
    EXPECT_EQ(entries.at("alpha").call_count, 2);
    EXPECT_EQ(entries.at("beta").call_count, 1);
    EXPECT_NEAR(entries.at("alpha").total_seconds, 0.25, 1e-10);

    reg.reset();
}

// ============================================================================
// ScopedTimer tests
// ============================================================================

TEST(Timer, ScopedTimerRecords) {
    auto& reg = TimerRegistry::instance();
    reg.reset();

    {
        ScopedTimer timer("scoped_test");
        // Minimal work
        volatile double x = 0;
        for (int i = 0; i < 1000; ++i) x += i;
        (void)x;
    }

    auto& entries = reg.entries();
    auto it = entries.find("scoped_test");
    ASSERT_NE(it, entries.end());
    EXPECT_EQ(it->second.call_count, 1);
    EXPECT_GE(it->second.total_seconds, 0.0);

    reg.reset();
}

TEST(Timer, ScopedTimerMeasuresTime) {
    auto& reg = TimerRegistry::instance();
    reg.reset();

    {
        ScopedTimer timer("sleep_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto& entries = reg.entries();
    EXPECT_GE(entries.at("sleep_test").total_seconds, 0.005);

    reg.reset();
}

// ============================================================================
// Logger tests
// ============================================================================

TEST(Logger, SetLevelDoesNotCrash) {
    auto& logger = Logger::instance();
    EXPECT_NO_THROW(logger.set_level(LogLevel::Debug));
    EXPECT_NO_THROW(logger.set_level(LogLevel::Info));
    EXPECT_NO_THROW(logger.set_level(LogLevel::Warning));
    EXPECT_NO_THROW(logger.set_level(LogLevel::Error));
    // Restore default
    logger.set_level(LogLevel::Error);  // suppress during tests
}

TEST(Logger, SetMPIRankDoesNotCrash) {
    auto& logger = Logger::instance();
    EXPECT_NO_THROW(logger.set_mpi_rank(0));
    EXPECT_NO_THROW(logger.set_mpi_rank(42));
    logger.set_mpi_rank(0);
}

TEST(Logger, LogDoesNotCrash) {
    auto& logger = Logger::instance();
    logger.set_level(LogLevel::Error);  // suppress output during test

    // These should not crash regardless of level
    EXPECT_NO_THROW(logger.debug("test", "debug message"));
    EXPECT_NO_THROW(logger.info("test", "info message"));
    EXPECT_NO_THROW(logger.warning("test", "warning message"));
    EXPECT_NO_THROW(logger.error("test", "error message"));
}

TEST(Logger, LogWithFields) {
    auto& logger = Logger::instance();
    logger.set_level(LogLevel::Error);

    std::map<std::string, std::string> fields = {
        {"scf_step", "5"},
        {"energy", "-15.0"},
        {"wall_s", "1.23"}
    };
    EXPECT_NO_THROW(logger.log(LogLevel::Info, "scf_iter", "step complete", fields));
}

// ============================================================================
// Constants sanity checks
// ============================================================================

TEST(Constants, BohrToAngstromConsistency) {
    EXPECT_NEAR(constants::bohr_to_angstrom * constants::angstrom_to_bohr,
                1.0, 1e-12);
}

TEST(Constants, HartreeToEvConsistency) {
    EXPECT_NEAR(constants::hartree_to_ev * constants::ev_to_hartree,
                1.0, 1e-10);
}

TEST(Constants, RydbergIsHalfHartree) {
    EXPECT_NEAR(constants::rydberg_to_ev, constants::hartree_to_ev / 2.0, 1e-6);
}

TEST(Constants, PiValue) {
    EXPECT_NEAR(constants::pi, 3.14159265358979323846, 1e-15);
    EXPECT_NEAR(constants::two_pi, 2.0 * constants::pi, 1e-15);
    EXPECT_NEAR(constants::four_pi, 4.0 * constants::pi, 1e-15);
}
