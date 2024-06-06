/**
 * @file tracker_factory.hpp
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

template <typename T>
class TrackerFactory {
public:
    static std::unique_ptr<T> Create(const int64_t& predict_duration);
};