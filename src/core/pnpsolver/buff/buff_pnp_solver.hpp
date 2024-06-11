/**
 * @file buff_pnp_solver.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-10
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include <optional>

#include "core/identifier/buff/buff.hpp"

#include "buff3d.hpp"

namespace auto_aim {

class BuffPnPSolver {
public:
    static std::optional<BuffPlate3d> Solve(
        const BuffPlate& buff, const rmcs_description::Tf& tf, const double& fx, const double& fy,
        const double& cx, const double& cy, const double& k1, const double& k2, const double& k3);

private:
    class StaticImpl;
};
} // namespace auto_aim