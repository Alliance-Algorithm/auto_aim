#pragma once
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <rmcs_core/msgs.hpp>

#include "core/identifier/armor/armor.hpp"
#include "core/identifier/identifier.hpp"

namespace auto_aim {

class ArmorIdentifier : public ArmorIdentifierInterface {
public:
    explicit ArmorIdentifier(const std::string& model_path);
    ~ArmorIdentifier();
    std::vector<ArmorPlate>
        Identify(const cv::Mat& img, const rmcs_core::msgs::RoboticColor& target_color);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace auto_aim
