/**
 * @file buff_identifier_impl.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-06
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include <optional>

#include "core/identifier/buff/buff.hpp"
#include "core/identifier/identifier.hpp"

class BuffIdentifier : public BuffIdentifierImpl {
public:
    explicit BuffIdentifier(const std::string& model_path);
    std::optional<BuffPlate> Identify(const cv::Mat& img) override;

private:
};
