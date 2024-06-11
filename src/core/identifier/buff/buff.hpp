/**
 * @file buff.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include <opencv2/opencv.hpp>
#include <rmcs_description/tf_description.hpp>

namespace auto_aim {
struct BuffPlate {
    // bl br tr tl
    std::vector<cv::Point2f> points;
};
} // namespace auto_aim
