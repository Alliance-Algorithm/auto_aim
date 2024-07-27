#pragma once
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <rmcs_msgs/robot_color.hpp>

#include "core/identifier/armor/armor.hpp"
#include "core/identifier/identifier.hpp"

namespace rmcs_auto_aim {

class ArmorIdentifier : public ArmorIdentifierInterface {
public:
    explicit ArmorIdentifier(const std::string& model_path);
    ~ArmorIdentifier();
    std::vector<ArmorPlate>
        Identify(const cv::Mat& img, const rmcs_msgs::RobotColor& target_color, int8_t blacklist);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace rmcs_auto_aim
