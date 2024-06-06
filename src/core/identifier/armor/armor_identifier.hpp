#pragma once
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <rmcs_core/msgs.hpp>

#include "core/identifier/armor/armor.hpp"
#include "core/identifier/identifier.hpp"

class ArmorIdentifier : public ArmorIdentifierImpl {
public:
    explicit ArmorIdentifier(const std::string& model_path);
    std::vector<ArmorPlate>
        Identify(const cv::Mat& img, const rmcs_core::msgs::RoboticColor& target_color) override;

private:
};