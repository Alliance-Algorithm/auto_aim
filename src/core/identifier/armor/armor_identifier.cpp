

#include "armor_identifier.hpp"
#include <rmcs_core/msgs.hpp>

class ArmorIdentifier::Impl {
public:
    explicit Impl(std::string model_path)
        : model_path_(std::move(model_path)) {}
    std::vector<ArmorPlate> Identify(const cv::Mat& img, rmcs_core::msgs::RoboticColor targetColor);

private:
    std::string model_path_;
};

std::vector<ArmorPlate>
    ArmorIdentifier::Identify(const cv::Mat& img, rmcs_core::msgs::RoboticColor targetColor) {
    return pImpl->Identify(img, targetColor);
}

ArmorIdentifier::ArmorIdentifier(const std::string& model_path)
    : pImpl(new Impl{model_path}) {}
ArmorIdentifier::~ArmorIdentifier() = default;