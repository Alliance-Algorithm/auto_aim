/**
 * @file armor_plate_3d.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include "core/identifier/armor/Armor.hpp"

struct ArmorPlate3d {
    ArmorID id;
    // GimbalGyro::Position position;
    // GimbalGyro::Rotation rotation;

    explicit ArmorPlate3d(ArmorID id)
        : id(id) {}
};