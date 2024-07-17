/**
 * @file debugger.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-07-04
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */

#pragma once
#include <memory>

#include <opencv2/core/mat.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/time.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "core/pnpsolver/armor/armor3d.hpp"

namespace rmcs_auto_aim {
class Debugger {
public:
    Debugger(const Debugger&)            = delete;
    Debugger& operator=(const Debugger&) = delete;

    static Debugger& getInstance() {
        static Debugger instance;
        return instance;
    }

    rclcpp::Node::SharedPtr getNode();

    void publish_raw_image(const cv::Mat& img, const rclcpp::Time& stamp);
    void publish_pnp_armor(const ArmorPlate3dWithNoFrame& armor);
    void publish_armors(const visualization_msgs::msg::MarkerArray& msg);
    // void publish_armors(const std::vector<rmcs_auto_aim::ArmorPlate>& armors, const cv::Mat&
    // img); void publish_3d_armors(const std::vector<rmcs_auto_aim::ArmorPlate3d>& armors); void
    // publish_raw_roi(const cv::Mat& img);

private:
    class Impl;

    Debugger();
    ~Debugger() = default;

    std::shared_ptr<Impl> pImpl_;
};
} // namespace rmcs_auto_aim