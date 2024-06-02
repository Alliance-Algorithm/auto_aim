#include "core/tracker/Target.hpp"

class Target::Impl {
public:
    Impl() = default;

    static rmcs_description::MuzzleLink::DirectionVector Predict(double sec) {
        // TODO -  Predict Impl
        (void)sec;
        return {};
    }
};

rmcs_description::MuzzleLink::DirectionVector Target::Predict(double sec) {
    return pImpl->Predict(sec);
}

Target::Target()
    : pImpl(new Impl{}) {}
Target::~Target() = default;
