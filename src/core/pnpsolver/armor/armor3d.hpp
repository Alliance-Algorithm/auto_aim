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

#include <utility>

#include <rmcs_description/tf_description.hpp>

#include "core/identifier/armor/armor.hpp"

namespace auto_aim {
struct ArmorPlate3d {
    ArmorID id;
    rmcs_description::OdomImu::Position position;
    rmcs_description::OdomImu::Rotation rotation;

    explicit ArmorPlate3d(
        ArmorID id, rmcs_description::OdomImu::Position position,
        rmcs_description::OdomImu::Rotation rotation)
        : id(id)
        , position(std::move(position))
        , rotation(std::move(rotation)) {}
};
} // namespace auto_aim
