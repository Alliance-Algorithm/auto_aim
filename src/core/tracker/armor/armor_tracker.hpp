/**
 * @file armor_tracker.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */

#pragma once
#include <chrono>
#include <memory>
#include <vector>

#include "core/pnpsolver/armor_plate_3d.hpp"
#include "core/tracker/Target.hpp"

class ArmorTracker {
public:
    explicit ArmorTracker(int64_t predict_duration);
    std::unique_ptr<Target> Update(
        const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};