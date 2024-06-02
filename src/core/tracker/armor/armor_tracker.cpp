

#include "armor_tracker.hpp"
#include <cstdint>

class ArmorTracker::Impl {
public:
    explicit Impl(int64_t predict_duration)
        : predict_duration_(predict_duration) {}

    std::unique_ptr<Target> Update(
        const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
        // TODO : Implement
        (void)predict_duration_;
        (void)armors;
        (void)timestamp;
        return {};
    }

private:
    int64_t predict_duration_;
};

std::unique_ptr<Target> ArmorTracker::Update(
    const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
    return pImpl->Update(armors, timestamp);
}

ArmorTracker::ArmorTracker(int64_t predict_duration)
    : pImpl(new Impl{predict_duration}) {}
