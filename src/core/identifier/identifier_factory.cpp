
#include "identifier_factory.hpp"
#include <memory>

template <typename T>
requires std::same_as<T, ArmorIdentifier> || std::same_as<T, BuffIdentifier>
std::unique_ptr<T> IdentifierFactory<T>::Create(const std::string& model_path) {
    return std::make_unique<T>(model_path);
}