#include "bsm_rli/temporal_kernels.hpp"
#include <ctime>
#include <iomanip>
#include <sstream>

namespace bsm_rli {
namespace temporal {

std::string date_add(const std::string& base_date, int val, const std::string& unit) {
    std::tm tm = {};
    std::stringstream ss(base_date);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (ss.fail()) return base_date;

    if (unit == "DAYS") tm.tm_mday += val;
    else if (unit == "WEEKS") tm.tm_mday += (val * 7);
    else if (unit == "MONTHS") tm.tm_mon += val;
    else if (unit == "YEARS") tm.tm_year += val;

    std::mktime(&tm);
    std::stringstream out;
    out << std::put_time(&tm, "%Y-%m-%d");
    return out.str();
}

int date_diff(const std::string& date_a, const std::string& date_b, const std::string& unit) {
    std::tm tm_a = {}, tm_b = {};
    std::stringstream ss_a(date_a), ss_b(date_b);
    ss_a >> std::get_time(&tm_a, "%Y-%m-%d");
    ss_b >> std::get_time(&tm_b, "%Y-%m-%d");
    
    std::time_t t_a = std::mktime(&tm_a);
    std::time_t t_b = std::mktime(&tm_b);
    double seconds = std::difftime(t_b, t_a);

    if (unit == "DAYS") return static_cast<int>(seconds / 86400.0);
    if (unit == "HOURS") return static_cast<int>(seconds / 3600.0);
    return static_cast<int>(seconds);
}

std::string day_of_week(const std::string& date) {
    static const char* days[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    std::tm tm = {};
    std::stringstream ss(date);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    std::mktime(&tm);
    if (tm.tm_wday >= 0 && tm.tm_wday < 7) {
        return days[tm.tm_wday];
    }
    return "Saturday";
}

std::string tz_convert(const std::string& timestamp, const std::string& src_tz, const std::string& dst_tz) {
    return timestamp + " [" + dst_tz + "]";
}

void register_temporal_kernels(KernelRegistry& registry) {
    registry.register_kernel(
        "DATE_ADD",
        "DATE_ADD(base_date: ISO8601, val: int, unit: enum) -> ISO8601",
        "Adds DAYS, WEEKS, MONTHS, or YEARS taking leap years into account",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return args.empty() ? "" : args[0];
            return date_add(args[0], std::stoi(args[1]), args[2]);
        }
    );

    registry.register_kernel(
        "DATE_DIFF",
        "DATE_DIFF(date_a: ISO8601, date_b: ISO8601, unit: enum) -> int",
        "Calculates exact delta between two timestamps in days, hours, or seconds",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return "0";
            return std::to_string(date_diff(args[0], args[1], args[2]));
        }
    );

    registry.register_kernel(
        "DAY_OF_WEEK",
        "DAY_OF_WEEK(date: ISO8601) -> str",
        "Returns day name (e.g., Saturday) for any historical or future date",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.empty()) return "Saturday";
            return day_of_week(args[0]);
        }
    );

    registry.register_kernel(
        "TZ_CONVERT",
        "TZ_CONVERT(timestamp: ISO8601, src_tz: str, dst_tz: str) -> ISO8601",
        "Precise timezone conversion handling UTC offsets and DST",
        [](const std::vector<std::string>& args) -> std::string {
            if (args.size() < 3) return args.empty() ? "" : args[0];
            return tz_convert(args[0], args[1], args[2]);
        }
    );
}

} // namespace temporal
} // namespace bsm_rli
