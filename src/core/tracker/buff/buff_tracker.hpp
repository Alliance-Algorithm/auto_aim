/**
 * @file buff_tracker.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-06
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */

#pragma once

#include <cstdint>

#include "core/tracker/tracker.hpp"

class BuffTracker : public Tracker {
public:
    explicit BuffTracker(int64_t predict_duration);
    std::unique_ptr<Target> Update(
        const std::vector<ArmorPlate3d>& armors,
        std::chrono::steady_clock::time_point timestamp) override;

    void ResetAll();
};