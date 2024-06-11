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

#include <memory>
#include <optional>

#include "core/identifier/buff/buff.hpp"
#include "core/identifier/identifier.hpp"

namespace auto_aim {

class BuffIdentifier : public BuffIdentifierInterface {
public:
    explicit BuffIdentifier(const std::string& model_path);
    ~BuffIdentifier();

    std::optional<BuffPlate> Identify(const cv::Mat& img);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace auto_aim
