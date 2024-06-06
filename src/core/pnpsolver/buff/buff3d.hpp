
#include <opencv2/core/types.hpp>

#include <rmcs_description/tf_description.hpp>

struct BuffPlate3d {
    rmcs_description::OdomImu::Position position;
    rmcs_description::OdomImu::Rotation rotation;
};