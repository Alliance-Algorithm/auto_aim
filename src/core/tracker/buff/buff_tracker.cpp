#include "buff_tracker.hpp"
#include "core/tracker/target.hpp"
#include <optional>

using namespace auto_aim;

class BuffTracker::Impl {
public:
    explicit Impl(int64_t predict_duration) { (void)predict_duration; }
    std::unique_ptr<TargetInterface> Update(
        const std::optional<BuffPlate3d>& buff, std::chrono::steady_clock::time_point timestamp) {
        (void)buff;
        (void)timestamp;
        (void)this;
        return {};
    }

    void ResetAll() {}
};

BuffTracker::BuffTracker(int64_t predict_duration)
    : pImpl_(new Impl{predict_duration}) {}

std::unique_ptr<TargetInterface> BuffTracker::Update(
    const std::optional<BuffPlate3d>& buff, std::chrono::steady_clock::time_point timestamp) {
    return pImpl_->Update(buff, timestamp);
}

void BuffTracker::ResetAll() { return pImpl_->ResetAll(); }

BuffTracker::~BuffTracker() = default;