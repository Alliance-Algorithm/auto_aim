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

#include <rmcs_description/tf_description.hpp>

class TargetInterface {
public:
    [[nodiscard]] virtual rmcs_description::OdomImu::Position Predict(double sec) const = 0;

private:
};
