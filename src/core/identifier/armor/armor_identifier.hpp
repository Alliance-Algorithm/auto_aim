/**
 * @file armor_identifier.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief Armor Identifier
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rmcs_core/msgs.hpp>

#include "Armor.hpp"

class ArmorIdentifier {
public:
    explicit ArmorIdentifier(const std::string& model_path);
    ~ArmorIdentifier();

    std::vector<ArmorPlate> Identify(const cv::Mat& img, rmcs_core::msgs::RoboticColor targetColor);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};