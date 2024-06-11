/**
 * @file identifier.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-02
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once
#include "core/identifier/armor/armor.hpp"
#include "core/identifier/buff/buff.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
#include <rmcs_core/msgs.hpp>

using namespace auto_aim;

template <typename ResultType>
class Identifier {
public:
    virtual ~Identifier(){};
    template <class... Args>
    ResultType Identify(Args... args);
};

typedef Identifier<std::vector<ArmorPlate>> ArmorIdentifierInterface;
typedef Identifier<std::optional<BuffPlate>> BuffIdentifierInterface;
