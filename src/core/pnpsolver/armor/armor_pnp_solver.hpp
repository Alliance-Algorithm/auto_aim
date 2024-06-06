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

class ArmorPnPSolver {
public:
    static std::vector<ArmorPlate3d> SolveAll(std::vector<ArmorPlate> armors);
};