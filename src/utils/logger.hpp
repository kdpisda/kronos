#pragma once
#include <string>
#include <sstream>
#include <map>

namespace kronos {

enum class LogLevel { Debug, Info, Warning, Error };

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    void set_mpi_rank(int rank);

    // Log a structured event as JSON line to stderr
    void log(LogLevel level, const std::string& event,
             const std::string& message,
             const std::map<std::string, std::string>& fields = {});

    // Convenience methods
    void debug(const std::string& event, const std::string& message,
               const std::map<std::string, std::string>& fields = {});
    void info(const std::string& event, const std::string& message,
              const std::map<std::string, std::string>& fields = {});
    void warning(const std::string& event, const std::string& message,
                 const std::map<std::string, std::string>& fields = {});
    void error(const std::string& event, const std::string& message,
               const std::map<std::string, std::string>& fields = {});

private:
    Logger() = default;
    LogLevel level_{LogLevel::Info};
    int mpi_rank_{0};

    std::string level_string(LogLevel level) const;
    std::string timestamp() const;  // ISO 8601
};

} // namespace kronos
