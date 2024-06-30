
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <eigen3/Eigen/Dense>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/utilities.hpp>

#include <fast_tf/impl/cast.hpp>
#include <hikcamera/image_capturer.hpp>
#include <rmcs_core/msgs.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>

#include "core/identifier/armor/armor_identifier.hpp"
#include "core/identifier/buff/buff_identifier.hpp"
#include "core/pnpsolver/armor/armor_pnp_solver.hpp"
#include "core/pnpsolver/buff/buff_pnp_solver.hpp"
#include "core/tracker/armor/armor_tracker.hpp"
#include "core/tracker/buff/buff_tracker.hpp"
#include "core/tracker/target.hpp"
#include "core/trajectory/trajectory_solvor.hpp"

namespace auto_aim {

class Controller
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    Controller()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {

        register_input("/predefined/update_count", update_count_);
        register_input("/tf", tf_);
        // register_input("/robot_color", color_);
        // register_input("/robot_id", robot_id_);
        // register_input("/auto_rune", buff_mode_);
        register_output(
            "/gimbal/auto_aim/control_direction", control_direction_, Eigen::Vector3d::Zero());

        exposure_time_           = get_parameter("exposure_time").as_int();
        armor_predict_duration_  = get_parameter("armor_predict_duration").as_int();
        buff_predict_duration_   = get_parameter("buff_predict_duration").as_int();
        yaw_error_               = get_parameter("yaw_error").as_double();
        pitch_error_             = get_parameter("pitch_error").as_double();
        gimbal_predict_duration_ = get_parameter("gimbal_predict_duration").as_int();
        armor_model_path_        = get_parameter("armor_model_path").as_string();
        buff_model_path_         = get_parameter("buff_model_path").as_string();
        fx                       = get_parameter("fx").as_double();
        fy                       = get_parameter("fy").as_double();
        cx                       = get_parameter("cx").as_double();
        cy                       = get_parameter("cy").as_double();
        k1                       = get_parameter("k1").as_double();
        k2                       = get_parameter("k2").as_double();
        k3                       = get_parameter("k3").as_double();
    }

    ~Controller() {
        if (gimbal_thread_.joinable())
            gimbal_thread_.join();
    }

    void update() override {
        if (*update_count_ == 0) {
            gimbal_thread_ = std::thread{[this]() {
                hikcamera::ImageCapturer::CameraProfile camera_profile;
                camera_profile.exposure_time = std::chrono::milliseconds(exposure_time_);
                camera_profile.gain          = 16.9807;
                if (robot_id_ == 7) {
                    // if (*robot_id_ == 7) {

                    camera_profile.invert_image = true;
                } else {
                    camera_profile.invert_image = false;
                }

                hikcamera::ImageCapturer img_capture(camera_profile);

                auto package_share_directory =
                    ament_index_cpp::get_package_share_directory("auto_aim");

                auto armor_identifier =
                    ArmorIdentifier(package_share_directory + armor_model_path_);
                auto buff_identifier = BuffIdentifier(package_share_directory + buff_model_path_);

                auto armor_tracker = ArmorTracker(armor_predict_duration_, *tf_);
                auto buff_tracker  = BuffTracker(buff_predict_duration_);

                auto buff_enabled = false;

                while (rclcpp::ok()) {
                    auto img       = img_capture.read();
                    auto timestamp = std::chrono::steady_clock::now();

                    if (!buff_enabled && buff_mode_) {
                        // if (!buff_enabled && *buff_mode_) {

                        buff_tracker.ResetAll();
                    }
                    buff_enabled = buff_mode_;
                    // buff_enabled = *buff_mode_;

                    if (!buff_enabled) {
                        auto armors = armor_identifier.Identify(img, color_);
                        // auto armors = armor_identifier.Identify(img, *color_);

                        auto armor3d =
                            ArmorPnPSolver::SolveAll(armors, *tf_, fx, fy, cx, cy, k1, k2, k3);

                        if (auto target = armor_tracker.Update(armor3d, timestamp)) {
                            timestamp_ = timestamp;
                            target_    = target.release();
                            break;
                        }

                    } else {
                        if (auto buff = buff_identifier.Identify(img)) {
                            if (auto buff3d =
                                    BuffPnPSolver::Solve(*buff, *tf_, fx, fy, cx, cy, k1, k2, k3)) {
                                if (auto target = buff_tracker.Update(*buff3d, timestamp)) {
                                    timestamp_ = timestamp;
                                    target_    = target.release();
                                    break;
                                }
                            }
                        }
                    }
                }
            }};
        }

        auto target = target_;
        if (!target_) {
            return;
        }

        using namespace std::chrono_literals;
        auto diff = std::chrono::steady_clock::now() - timestamp_;
        if (diff > std::chrono::milliseconds(gimbal_predict_duration_)) {
            *control_direction_ = Eigen::Vector3d::Zero();
            return;
        }

        double fly_time = 0;
        for (int i = 5; i-- > 0;) {
            auto pos = target->Predict(
                static_cast<std::chrono::duration<double>>(diff).count() + fly_time + 0.05);
            auto aiming_direction = *trajectory_.GetShotVector(pos, 27.0, fly_time);

            auto yaw_axis = fast_tf::cast<rmcs_description::PitchLink>(
                                rmcs_description::OdomImu::DirectionVector(0, 0, 1), *tf_)
                                ->normalized();
            auto pitch_axis = fast_tf::cast<rmcs_description::PitchLink>(
                                  rmcs_description::OdomImu::DirectionVector(0, 1, 0), *tf_)
                                  ->normalized();
            auto delta_yaw   = Eigen::AngleAxisd{yaw_error_, yaw_axis};
            auto delta_pitch = Eigen::AngleAxisd{pitch_error_, pitch_axis};
            aiming_direction = delta_pitch * (delta_yaw * (aiming_direction));
            if (i == 0) {
                *control_direction_ = aiming_direction;
            }
        }
    }

private:
    // InputInterface<rmcs_core::msgs::RoboticColor> color_;
    InputInterface<rmcs_description::Tf> tf_;
    InputInterface<size_t> update_count_;
    // InputInterface<uint8_t> robot_id_;
    // InputInterface<bool> buff_mode_;

    rmcs_core::msgs::RoboticColor color_ = rmcs_core::msgs::RoboticColor::Blue;
    uint8_t robot_id_                    = 7;
    bool buff_mode_                      = false;

    OutputInterface<Eigen::Vector3d> control_direction_;

    std::chrono::steady_clock::time_point timestamp_;
    std::thread gimbal_thread_;

    int64_t gimbal_predict_duration_;
    int64_t armor_predict_duration_;
    int64_t buff_predict_duration_;
    int64_t exposure_time_;

    TrajectorySolver trajectory_{};
    TargetInterface* target_;

    std::string armor_model_path_;
    std::string buff_model_path_;

    double pitch_error_;
    double yaw_error_;

    double fx, fy, cx, cy, k1, k2, k3;
};

}; // namespace auto_aim

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(auto_aim::Controller, rmcs_executor::Component)