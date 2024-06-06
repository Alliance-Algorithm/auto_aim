/**
 * @file identifier.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once
#include "core/identifier/armor/armor.hpp"
#include "core/identifier/buff/buff.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
#include <rmcs_core/msgs.hpp>

template <typename ResultType, typename... Args>
class Identifier {
public:
    virtual ~Identifier(){};
    virtual ResultType Identify(Args... args) = 0;
};

typedef Identifier<std::vector<ArmorPlate>, const cv::Mat&, const rmcs_core::msgs::RoboticColor&>
    ArmorIdentifierImpl;
typedef Identifier<std::optional<BuffPlate>, const cv::Mat&> BuffIdentifierImpl;
