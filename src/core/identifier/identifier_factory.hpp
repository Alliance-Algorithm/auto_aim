/**
 * @file identifier_factory.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-06
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once
#include <concepts>
#include <memory>
#include <string>

#include <rmcs_core/msgs.hpp>

#include "core/identifier/armor/armor_identifier.hpp"
#include "core/identifier/buff/buff_identifier.hpp"

template <typename T>
requires std::same_as<T, ArmorIdentifier> || std::same_as<T, BuffIdentifier>
class IdentifierFactory {
public:
    static std::unique_ptr<T> Create(const std::string& model_path);
};
