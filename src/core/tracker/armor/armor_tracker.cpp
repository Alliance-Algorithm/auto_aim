#include "core/tracker/target.hpp"
#include <cstdint>

#include "armor_tracker.hpp"

using namespace auto_aim;

class ArmorTracker::Impl {
public:
    explicit Impl(int64_t predict_duration) { (void)predict_duration; }
    std::unique_ptr<TargetInterface> Update(
        const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
        (void)armors;
        (void)timestamp;
        (void)this;
        return {};
    }
};

ArmorTracker::ArmorTracker(int64_t predict_duration)
    : pImpl_(new Impl{predict_duration}) {}

std::unique_ptr<TargetInterface> ArmorTracker::Update(
    const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
    return pImpl_->Update(armors, timestamp);
}

ArmorTracker::~ArmorTracker() = default;
