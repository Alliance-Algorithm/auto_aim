#include "buff_tracker.hpp"

BuffTracker::BuffTracker(int64_t predict_duration) { (void)predict_duration; }

std::unique_ptr<Target> BuffTracker::Update(
    const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
    (void)armors;
    (void)timestamp;
    return {};
}
