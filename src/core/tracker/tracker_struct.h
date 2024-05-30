#pragma once
/*
Creation Date: 2023/08/09
Latest Update: 2023/08/09
Developer(s): 22-Qzh,23-Ftz
(C)Copyright: NJUST.Alliance - All rights reserved
Header Functions:
- Target的接口
*/

#include <rmcs_description/tf_description.hpp>

class TargetInterface {
public:
    virtual ~TargetInterface() = default;
    [[nodiscard]] virtual rmcs_description::MuzzleLink::DirectionVector
        Predict(double sec) const = 0;
};