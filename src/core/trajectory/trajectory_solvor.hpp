#pragma once

#include <cmath>
#include <tuple>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <fast_tf/rcl.hpp>
#include <rmcs_description/tf_description.hpp>

#include "core/tracker/target.hpp"

class TrajectorySolver {
public:
    /*! 获取射击角度
     * \return 返回云台偏移量，格式为tuple[yaw, pitch]，单位弧度制，遵循右手定则。
     */
    static std::tuple<double, double> GetShotAngle(
        TargetInterface& target, const double& speed, bool predict_movement = true,
        double time_shift = 0) {
        std::tuple<double, double> result;
        auto& [yaw, pitch] = result;
        double fly_time    = 0;
        rmcs_description::OdomImu::Position pos;
        GetShotAngleInternal(target.Predict(time_shift), speed, yaw, pitch, fly_time);
        if (predict_movement) {
            GetShotAngleInternal(
                pos = target.Predict(time_shift + fly_time), speed, yaw, pitch, fly_time);
        }

        return result;
    }

    static void GetShotAngleInternal(
        const rmcs_description::OdomImu::Position& target_pos, const double speed, double& yaw,
        double& pitch, double& fly_time) {
        auto shotVec = GetShotVector(target_pos, speed, fly_time);

        yaw = atan2(shotVec->y(), shotVec->x());
        pitch =
            -atan2(shotVec->z(), sqrt(shotVec->y() * shotVec->y() + shotVec->x() + shotVec->x()));
    }

    [[nodiscard]] static rmcs_description::MuzzleLink::DirectionVector GetShotVector(
        const rmcs_description::OdomImu::Position& target_pos, const double speed,
        double& fly_time) {
        // 不考虑空气阻力

        const double& x = target_pos->x();
        const double& y = target_pos->y();
        const double& z = target_pos->z();

        double yaw   = atan2(y, x);
        double pitch = 0;

        double a = speed * speed; // v0 ^ 2
        double b = a * a;         // v0 ^ 4
        double c = x * x + y * y; // xt ^ 2
        double d = c * c;         // xt ^ 4
        double e = G * G;         // g ^ 2

        double xt = sqrt(c);      // target horizontal distance

        double f = b * d * (b - e * c - 2 * G * a * z);
        if (f >= 0) {
            pitch = -atan((b * c - sqrt(f)) / (G * a * c * xt));
        }

        auto result = rmcs_description::MuzzleLink::DirectionVector{
            cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), -sin(pitch)};

        fly_time = xt / (cos(pitch) * speed);

        return result;
    }

private:
    constexpr const static double G = 9.80665;
};
