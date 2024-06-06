#include "buff_identifier.hpp"

// TODO

BuffIdentifier::BuffIdentifier(const std::string& model_path) { (void)model_path; }

std::optional<BuffPlate> BuffIdentifier::Identify(const cv::Mat& img) {
    (void)img;
    return {};
}