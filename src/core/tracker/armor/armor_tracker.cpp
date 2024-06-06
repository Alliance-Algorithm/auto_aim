#include "armor_tracker.hpp"

ArmorTracker::ArmorTracker(int64_t predict_duration) { (void)predict_duration; }

std::unique_ptr<Target> ArmorTracker::Update(
    const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
    (void)armors;
    (void)timestamp;
    return {};
}
