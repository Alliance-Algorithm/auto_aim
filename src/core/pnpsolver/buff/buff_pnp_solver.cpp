#include "buff_pnp_solver.hpp"
#include <fast_tf/impl/cast.hpp>
#include <rmcs_description/tf_description.hpp>

using namespace auto_aim;

class BuffPnPSolver::StaticImpl {
public:
    static std::optional<BuffPlate3d> Solve(
        const BuffPlate& buff, const rmcs_description::Tf& tf, const double& fx, const double& fy,
        const double& cx, const double& cy, const double& k1, const double& k2, const double& k3) {
        cv::Mat rvec, tvec;

        if (cv::solvePnP(
                BuffObjectPoints, buff.points,
                (cv::Mat)(cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1),
                (cv::Mat)(cv::Mat_<double>(1, 5) << k1, k2, 1, 1, k3), rvec, tvec, false,
                cv::SOLVEPNP_IPPE)) {
            Eigen::Vector3d position = {
                tvec.at<double>(2), -tvec.at<double>(0), -tvec.at<double>(1)};
            position = position / 1000.0;
            if (position.norm() > MaxArmorDistance)
                return std::nullopt;

            Eigen::Vector3d rvec_eigen = {
                rvec.at<double>(2), -rvec.at<double>(0), -rvec.at<double>(1)};
            Eigen::Quaterniond rotation = Eigen::Quaterniond{
                Eigen::AngleAxisd{rvec_eigen.norm(), rvec_eigen.normalized()}
            };

            return BuffPlate3d{
                fast_tf::cast<rmcs_description::OdomImu>(
                    rmcs_description::CameraLink::Position{position}, tf),
                fast_tf::cast<rmcs_description::OdomImu>(
                    rmcs_description::CameraLink::Rotation{rotation}, tf)};
        }
        return std::nullopt;
    }

private:
    inline static constexpr double BuffPlateHeightL = 372, BuffPlateHeightR = 320,
                                   BuffPlateWidth                 = 317;
    inline static const std::vector<cv::Point3d> BuffObjectPoints = {
        cv::Point3d{0, -0.5 * BuffPlateHeightL, -0.5 * BuffPlateWidth},
        cv::Point3d{0, -0.5 * BuffPlateHeightR,  0.5 * BuffPlateWidth},
        cv::Point3d{0,  0.5 * BuffPlateHeightR,  0.5 * BuffPlateWidth},
        cv::Point3d{0,  0.5 * BuffPlateHeightL, -0.5 * BuffPlateWidth},
    };
    inline static const double MaxArmorDistance = 15.0;
};

std::optional<BuffPlate3d> BuffPnPSolver::Solve(
    const BuffPlate& buff, const rmcs_description::Tf& tf, const double& fx, const double& fy,
    const double& cx, const double& cy, const double& k1, const double& k2, const double& k3) {
    return BuffPnPSolver::StaticImpl::Solve(buff, tf, fx, fy, cx, cy, k1, k2, k3);
}
