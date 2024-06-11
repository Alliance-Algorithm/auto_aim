
#include <opencv2/core/types.hpp>

#include <rmcs_description/tf_description.hpp>

namespace auto_aim {
struct BuffPlate3d {
    rmcs_description::OdomImu::Position position;
    rmcs_description::OdomImu::Rotation rotation;
    explicit BuffPlate3d(
        rmcs_description::OdomImu::Position position, rmcs_description::OdomImu::Rotation rotation)
        : position(std::move(position))
        , rotation(std::move(rotation)) {}
};
} // namespace auto_aim