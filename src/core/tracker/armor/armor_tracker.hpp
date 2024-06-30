/**
 * @file armor_tracker.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-06
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

#include "core/pnpsolver/armor/armor3d.hpp"
#include "core/tracker/target.hpp"
#include "core/tracker/tracker.hpp"

namespace auto_aim {
class ArmorTracker : public TrackerInterface {
public:
    explicit ArmorTracker(int64_t predict_duration, const rmcs_description::Tf& tf);
    ~ArmorTracker();

    std::unique_ptr<TargetInterface> Update(
        const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // namespace auto_aim
