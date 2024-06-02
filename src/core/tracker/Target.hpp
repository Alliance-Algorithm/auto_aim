/**
 * @file Target.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief Target Interface
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include <memory>
#include <rmcs_description/tf_description.hpp>

class Target {
public:
    Target();
    ~Target();
    [[nodiscard]] rmcs_description::MuzzleLink::DirectionVector Predict(double sec);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};