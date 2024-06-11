/**
 * @file tracker.hpp
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

#include "core/tracker/target.hpp"

namespace auto_aim {
class TrackerInterface {
public:
    virtual ~TrackerInterface() {}

    template <class... Args>
    std::unique_ptr<TargetInterface> Update(Args... args);
};
} // namespace auto_aim
