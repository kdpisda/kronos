#include "utils/logger.hpp"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace kronos {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level) {
    level_ = level;
}

void Logger::set_mpi_rank(int rank) {
    mpi_rank_ = rank;
}

void Logger::log(LogLevel level, const std::string& event,
                 const std::string& message,
                 const std::map<std::string, std::string>& fields) {
    // Only log if level >= configured level
    if (static_cast<int>(level) < static_cast<int>(level_)) {
        return;
    }

    // Only rank 0 logs by default (for MPI)
    if (mpi_rank_ != 0) {
        return;
    }

    // Build JSON line manually to avoid external dependencies
    std::ostringstream oss;
    oss << "{";
    oss << "\"timestamp\":\"" << timestamp() << "\"";
    oss << ",\"level\":\"" << level_string(level) << "\"";
    oss << ",\"event\":\"" << event << "\"";
    oss << ",\"mpi_rank\":" << mpi_rank_;

    // Escape message string for JSON (handle quotes and backslashes)
    oss << ",\"message\":\"";
    for (char c : message) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:   oss << c;      break;
        }
    }
    oss << "\"";

    // Append extra fields
    for (const auto& [key, value] : fields) {
        oss << ",\"" << key << "\":\"";
        for (char c : value) {
            switch (c) {
                case '"':  oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\n': oss << "\\n";  break;
                case '\r': oss << "\\r";  break;
                case '\t': oss << "\\t";  break;
                default:   oss << c;      break;
            }
        }
        oss << "\"";
    }

    oss << "}\n";

    std::cerr << oss.str();
}

void Logger::debug(const std::string& event, const std::string& message,
                   const std::map<std::string, std::string>& fields) {
    log(LogLevel::Debug, event, message, fields);
}

void Logger::info(const std::string& event, const std::string& message,
                  const std::map<std::string, std::string>& fields) {
    log(LogLevel::Info, event, message, fields);
}

void Logger::warning(const std::string& event, const std::string& message,
                     const std::map<std::string, std::string>& fields) {
    log(LogLevel::Warning, event, message, fields);
}

void Logger::error(const std::string& event, const std::string& message,
                   const std::map<std::string, std::string>& fields) {
    log(LogLevel::Error, event, message, fields);
}

std::string Logger::level_string(LogLevel level) const {
    switch (level) {
        case LogLevel::Debug:   return "debug";
        case LogLevel::Info:    return "info";
        case LogLevel::Warning: return "warning";
        case LogLevel::Error:   return "error";
    }
    return "unknown";
}

std::string Logger::timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&time_t_now, &tm_buf);

    // Format as ISO 8601: YYYY-MM-DDTHH:MM:SSZ
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S") << "Z";
    return oss.str();
}

} // namespace kronos
