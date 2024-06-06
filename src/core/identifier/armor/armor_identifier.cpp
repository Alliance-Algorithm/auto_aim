#include "armor_identifier.hpp"
#include <string>

// TODO

ArmorIdentifier::ArmorIdentifier(const std::string& model_path) { (void)model_path; }

std::vector<ArmorPlate> ArmorIdentifier::Identify(
    const cv::Mat& img, const rmcs_core::msgs::RoboticColor& target_color) {
    (void)img;
    (void)target_color;
    return {};
}