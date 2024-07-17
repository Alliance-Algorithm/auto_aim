#include <cmath>

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include <fast_tf/impl/cast.hpp>
#include <rmcs_description/tf_description.hpp>

#include "armor_pnp_solver.hpp"
#include "core/pnpsolver/armor/armor3d.hpp"

using namespace rmcs_auto_aim;

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

                auto yaw   = atan(position.y() / position.x());
                auto depth = position.norm();
                auto pitch = asin(position.z() / depth);

                auto corrected_depth = correct(yaw, depth, armor.is_large_armor);
                auto corrected_pos   = rmcs_description::CameraLink::Position{
                    corrected_depth * cos(pitch) * cos(yaw),
                    corrected_depth * cos(pitch) * sin(yaw), corrected_depth * sin(pitch)};

                armors3d.emplace_back(
                    armor.id, fast_tf::cast<rmcs_description::OdomImu>(corrected_pos, tf),
                    fast_tf::cast<rmcs_description::OdomImu>(
                        rmcs_description::CameraLink::Rotation{rotation}, tf));
            } else {
                continue;
            }
        }

        return armors3d;
    }

    static ArmorPlate3dWithNoFrame Solve(
        const ArmorPlate& armor, const double& fx, const double& fy, const double& cx,
        const double& cy, const double& k1, const double& k2, const double& k3) {

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
                return {};
            }

            Eigen::Vector3d rvec_eigen = {
                rvec.at<double>(2), -rvec.at<double>(0), -rvec.at<double>(1)};
            Eigen::Quaterniond rotation = Eigen::Quaterniond{
                Eigen::AngleAxisd{rvec_eigen.norm(), rvec_eigen.normalized()}
            };

            return {armor.id, position, rotation};
        }

        return {};
    }

private:
    static double correct(const double& yaw, const double& depth, const bool& is_large_armor) {
        if (is_large_armor) {
            auto& a1 = LargeArmorCorrectFactor[0];
            auto& a2 = LargeArmorCorrectFactor[1];
            auto& b1 = LargeArmorCorrectFactor[2];
            auto& b2 = LargeArmorCorrectFactor[3];
            auto& c1 = LargeArmorCorrectFactor[4];
            auto& c2 = LargeArmorCorrectFactor[5];
            auto& d  = LargeArmorCorrectFactor[6];
            auto& e1 = LargeArmorCorrectFactor[7];
            auto& e2 = LargeArmorCorrectFactor[8];
            auto& e3 = LargeArmorCorrectFactor[9];
            auto& e4 = LargeArmorCorrectFactor[10];
            auto& e5 = LargeArmorCorrectFactor[11];
            auto& e6 = LargeArmorCorrectFactor[12];
            auto& f1 = LargeArmorCorrectFactor[13];
            auto& f2 = LargeArmorCorrectFactor[14];

            return c1 * -cos(yaw) + c2 * depth + e1 * depth * -cos(yaw)
                 + e2 * depth * -cos(yaw) * -cos(yaw) + e3 * depth * depth * -cos(yaw)
                 + e4 * depth * depth * -cos(yaw) * -cos(yaw)
                 + e5 * depth * depth * -cos(yaw) * -cos(yaw) * -cos(yaw)
                 + e6 * depth * depth * depth * -cos(yaw) * -cos(yaw) + d
                 + a1 * -cos(yaw) * -cos(yaw) * -cos(yaw) + b1 * -cos(yaw) * -cos(yaw)
                 + a2 * depth * depth * depth + b2 * depth * depth
                 + f1 * -cos(yaw) * -cos(yaw) * -cos(yaw) * -cos(yaw)
                 + f2 * depth * depth * depth * depth;
        } else {
            auto& a1 = NormalArmorCorrectFactor[0];
            auto& a2 = NormalArmorCorrectFactor[1];
            auto& b1 = NormalArmorCorrectFactor[2];
            auto& b2 = NormalArmorCorrectFactor[3];
            auto& c1 = NormalArmorCorrectFactor[4];
            auto& c2 = NormalArmorCorrectFactor[5];
            auto& d  = NormalArmorCorrectFactor[6];
            auto& e1 = NormalArmorCorrectFactor[7];
            auto& e2 = NormalArmorCorrectFactor[8];
            auto& e3 = NormalArmorCorrectFactor[9];
            auto& e4 = NormalArmorCorrectFactor[10];
            auto& e5 = NormalArmorCorrectFactor[11];
            auto& e6 = NormalArmorCorrectFactor[12];
            auto& f1 = NormalArmorCorrectFactor[13];
            auto& f2 = NormalArmorCorrectFactor[14];

            return c1 * -cos(yaw) + c2 * depth + e1 * depth * -cos(yaw)
                 + e2 * depth * -cos(yaw) * -cos(yaw) + e3 * depth * depth * -cos(yaw)
                 + e4 * depth * depth * -cos(yaw) * -cos(yaw)
                 + e5 * depth * depth * -cos(yaw) * -cos(yaw) * -cos(yaw)
                 + e6 * depth * depth * depth * -cos(yaw) * -cos(yaw) + d
                 + a1 * -cos(yaw) * -cos(yaw) * -cos(yaw) + b1 * -cos(yaw) * -cos(yaw)
                 + a2 * depth * depth * depth + b2 * depth * depth
                 + f1 * -cos(yaw) * -cos(yaw) * -cos(yaw) * -cos(yaw)
                 + f2 * depth * depth * depth * depth;
        }
    }

    inline static const double LargeArmorCorrectFactor[] = {
        -5.75499912e+05, 6.24999075e-01,  8.55156294e+05,  -1.31784196e+04, -5.64802294e+05,
        -3.62809911e+02, 7.35939798e+02,  -3.72008321e+02, 4.03873153e+04,  -4.12562293e+04,
        1.40472334e+04,  -5.89194756e-01, 1.45248168e+05,  -3.82473750e-03, 1.39897763e+05};
    inline static const double NormalArmorCorrectFactor[] = {
        9.20311964e+03,  9.13950159e-01,  -6.07520262e+03, -5.74549783e+03, -9.09107445e+02,
        -1.91307592e+02, 4.05636080e+02,  -2.13223192e+02, 1.77120913e+04,  -1.82083300e+04,
        6.24168392e+03,  -8.96730075e-01, -3.63700027e+03, -1.70822160e-03, 1.41813817e+03};

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

ArmorPlate3dWithNoFrame ArmorPnPSolver::Solve(
    const ArmorPlate& armor, const double& fx, const double& fy, const double& cx, const double& cy,
    const double& k1, const double& k2, const double& k3) {
    return ArmorPnPSolver::StaticImpl::Solve(armor, fx, fy, cx, cy, k1, k2, k3);
}