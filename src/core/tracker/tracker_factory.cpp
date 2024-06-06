#include "tracker_factory.hpp"
#include <memory>

template <typename T>
std::unique_ptr<T> TrackerFactory<T>::Create(const int64_t& predict_duration) {
    return std::make_unique<T>(predict_duration);
}