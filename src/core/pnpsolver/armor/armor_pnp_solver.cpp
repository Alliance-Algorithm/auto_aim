#include <fast_tf/impl/cast.hpp>
#include <opencv2/core/mat.hpp>

#include <opencv2/opencv.hpp>
#include <rmcs_description/tf_description.hpp>

#include "armor_pnp_solver.hpp"

using namespace auto_aim;

class ArmorPnPSolver::StaticImpl {
public:
    static std::vector<ArmorPlate3d> SolveAll(
        const std::vector<ArmorPlate>& armors, const rmcs_description::Tf& tf, const double& fx,
        const double& fy, const double& cx, const double& cy, const double& k1, const double& k2,
        const double& k3) {
        std::vector<ArmorPlate3d> armors3d;

        for (const auto& armor : armors) {
            cv::Mat rvec, tvec;
            auto& objectPoints =
                armor.is_large_armor ? LargeArmorObjectPoints : NormalArmorObjectPoints;
            if (cv::solvePnP(
                    objectPoints, armor.points,
                    (cv::Mat)(cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1),
                    (cv::Mat)(cv::Mat_<double>(1, 5) << k1, k2, 0, 0, k3), rvec, tvec, false,
                    cv::SOLVEPNP_IPPE)) {

                Eigen::Vector3d position = {
                    tvec.at<double>(2), -tvec.at<double>(0), -tvec.at<double>(1)};
                position = position / 1000.0;
                if (position.norm() > MaxArmorDistance) {
                    continue;
                }

                Eigen::Vector3d rvec_eigen = {
                    rvec.at<double>(2), -rvec.at<double>(0), -rvec.at<double>(1)};
                Eigen::Quaterniond rotation = Eigen::Quaterniond{
                    Eigen::AngleAxisd{rvec_eigen.norm(), rvec_eigen.normalized()}
                };

                armors3d.emplace_back(
                    armor.id,
                    fast_tf::cast<rmcs_description::OdomImu>(
                        rmcs_description::CameraLink::Position{position}, tf),
                    fast_tf::cast<rmcs_description::OdomImu>(
                        rmcs_description::CameraLink::Rotation{rotation}),
                    tf);
            } else {
                continue;
            }
        }

        return armors3d;
    }

private:
    inline constexpr static const double MaxArmorDistance = 15.0;

    inline constexpr static const double NormalArmorWidth = 134, NormalArmorHeight = 56,
                                         LargerArmorWidth = 230, LargerArmorHeight = 56;
    inline static const std::vector<cv::Point3d> LargeArmorObjectPoints = {
        cv::Point3d(-0.5 * LargerArmorWidth, 0.5 * LargerArmorHeight, 0.0f),
        cv::Point3d(-0.5 * LargerArmorWidth, -0.5 * LargerArmorHeight, 0.0f),
        cv::Point3d(0.5 * LargerArmorWidth, -0.5 * LargerArmorHeight, 0.0f),
        cv::Point3d(0.5 * LargerArmorWidth, 0.5 * LargerArmorHeight, 0.0f)};
    inline static const std::vector<cv::Point3d> NormalArmorObjectPoints = {
        cv::Point3d(-0.5 * NormalArmorWidth, 0.5 * NormalArmorHeight, 0.0f),
        cv::Point3d(-0.5 * NormalArmorWidth, -0.5 * NormalArmorHeight, 0.0f),
        cv::Point3d(0.5 * NormalArmorWidth, -0.5 * NormalArmorHeight, 0.0f),
        cv::Point3d(0.5 * NormalArmorWidth, 0.5 * NormalArmorHeight, 0.0f)};
};

std::vector<ArmorPlate3d> ArmorPnPSolver::SolveAll(
    const std::vector<ArmorPlate>& armors, const rmcs_description::Tf& tf, const double& fx,
    const double& fy, const double& cx, const double& cy, const double& k1, const double& k2,
    const double& k3) {
    return ArmorPnPSolver::StaticImpl::SolveAll(armors, tf, fx, fy, cx, cy, k1, k2, k3);
}