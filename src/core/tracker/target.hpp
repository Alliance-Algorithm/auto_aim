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

class Target {
public:
    // class Impl;
    // explicit Target(std::unique_ptr<class Impl> impl);
    // ~Target();
    [[nodiscard]] virtual rmcs_description::OdomImu::Position Predict(double sec) = 0;

private:
    // std::unique_ptr<Impl> pImpl;
};

// class Target::Impl {
// public:
//     virtual ~Impl() = default;

//     virtual rmcs_description::OdomImu::Position Predict(double sec) = 0;
// };

// rmcs_description::OdomImu::Position Target::Predict(double sec) { return pImpl->Predict(sec); }

// Target::Target(std::unique_ptr<Impl> impl)
//     : pImpl(std::move(impl)) {}

// Target::~Target() = default;
