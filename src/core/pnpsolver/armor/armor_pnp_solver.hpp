/**
 * @file armor_pnp_solver.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once
#include <vector>

#include "core/identifier/armor/armor.hpp"
#include "core/pnpsolver/armor/armor3d.hpp"

namespace auto_aim {

class ArmorPnPSolver {
public:
    static std::vector<ArmorPlate3d> SolveAll(
        const std::vector<ArmorPlate>& armors, const rmcs_description::Tf& tf, const double& fx,
        const double& fy, const double& cx, const double& cy, const double& k1, const double& k2,
        const double& k3);

private:
    class StaticImpl;
};
} // namespace auto_aim