/**
 * @file track.hpp
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
#include <memory>

#include "core/pnpsolver/armor/armor3d.hpp"
#include "core/tracker/target.hpp"

class Tracker {
public:
    virtual ~Tracker() {}
    virtual std::unique_ptr<Target> Update(
        const std::vector<ArmorPlate3d>& armors,
        std::chrono::steady_clock::time_point timestamp) = 0;
};