#include <cstdint>
#include <map>

#include <fast_tf/impl/cast.hpp>

#include "armor_tracker.hpp"
#include "core/tracker/target.hpp"

using namespace auto_aim;

class EKF {
public:
    EKF() {
        Eigen::DiagonalMatrix<double, 9> p;
        p.setIdentity();
        P_ = p;
    };

    void Predict(double t) {
        Eigen::MatrixXd F = jacobian_f(x_, t), Q = get_Q(t);

        x_ = f(x_, t);
        P_ = F * P_ * F.transpose() + Q;
    }

    [[nodiscard]] Eigen::VectorXd PredictConst(double t) const { return f(x_, t); }

    void Update(const Eigen::VectorXd& z) {
        Eigen::MatrixXd H = jacobian_h(x_), R = get_R(z);
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(9, 9);

        Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();
        x_                = x_ + K * (z - h(x_));
        P_                = (I - K * H) * P_;

        /*std::cout << "ababab\n";
        std::cout << K;
        std::this_thread::sleep_for(std::chrono::seconds(100));*/
    }

    Eigen::Matrix<double, 9, 1> x_;
    Eigen::Matrix<double, 9, 9> P_;

private:
    static constexpr double sigma2_q_xyz_ = 20.0;
    static constexpr double sigma2_q_yaw_ = 100.0;
    static constexpr double sigma2_q_r_   = 800.0;
    static constexpr double r_xyz_factor_ = 0.05;
    static constexpr double r_yaw_        = 0.02;

    // f - Process function
    static Eigen::VectorXd f(const Eigen::VectorXd& x, double dt) {
        Eigen::VectorXd x_new = x;
        x_new(0) += x(1) * dt;
        x_new(2) += x(3) * dt;
        x_new(4) += x(5) * dt;
        x_new(6) += x(7) * dt;
        return x_new;
    }

    // J_f - Jacobian of process function
    static Eigen::MatrixXd jacobian_f(const Eigen::VectorXd& x, double dt) {
        (void)x;
        Eigen::MatrixXd f(9, 9);
        f << 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
        return f;
    };

    // h - Observation function
    static Eigen::VectorXd h(const Eigen::VectorXd& x) {
        Eigen::VectorXd z(4);
        double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
        z(0) = xc - r * cos(yaw); // xa
        z(1) = yc - r * sin(yaw); // ya
        z(2) = x(4);              // za
        z(3) = x(6);              // yaw
        return z;
    };

    // jacobian_h - Jacobian of process function
    static Eigen::MatrixXd jacobian_h(const Eigen::VectorXd& x) {
        Eigen::MatrixXd h(4, 9);
        double yaw = x(6), r = x(8);
        // clang-format off
        //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
        h << 1, 0, 0, 0, 0, 0, r* sin(yaw), 0, -cos(yaw),
                0, 0, 1, 0, 0, 0, -r * cos(yaw), 0, -sin(yaw),
                0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0;
        // clang-format on
        return h;
    };

    // Q - process noise covariance matrixEigen::Vector3d
    static Eigen::MatrixXd get_Q(double dt) {
        Eigen::MatrixXd q(9, 9);
        double t = dt, x = sigma2_q_xyz_, y = sigma2_q_yaw_, r = sigma2_q_r_;
        double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
        double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
        double q_r = pow(t, 4) / 4 * r;
        // clang-format off
        //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
        q << q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0,
                q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0, 0,
                0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0,
                0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0,
                0, 0, 0, 0, q_x_x, q_x_vx, 0, 0, 0,
                0, 0, 0, 0, q_x_vx, q_vx_vx, 0, 0, 0,
                0, 0, 0, 0, 0, 0, q_y_y, q_y_vy, 0,
                0, 0, 0, 0, 0, 0, q_y_vy, q_vy_vy, 0,
                0, 0, 0, 0, 0, 0, 0, 0, q_r;
        // clang-format on
        return q;
    };

    // R - measurement noise covariance matrix
    static Eigen::DiagonalMatrix<double, 4> get_R(const Eigen::VectorXd& z) {
        Eigen::DiagonalMatrix<double, 4> r;
        double x = r_xyz_factor_;
        r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw_;
        return r;
    };

    // P - error estimate covariance matrix
    // static Eigen::DiagonalMatrix<double, 9> get_P() {
    //}
};

class ArmorTracker::Impl {
public:
    explicit Impl(int64_t predict_duration, const rmcs_description::Tf& tf)
        : predict_duration_(predict_duration)
        , tf_(tf) {
        tracker_map_[ArmorID::Hero]        = {};
        tracker_map_[ArmorID::Engineer]    = {};
        tracker_map_[ArmorID::InfantryIII] = {};
        tracker_map_[ArmorID::InfantryIV]  = {};
        tracker_map_[ArmorID::InfantryV]   = {};
        tracker_map_[ArmorID::Sentry]      = {};
        tracker_map_[ArmorID::Outpost]     = {};
    }

    std::unique_ptr<TargetInterface> Update(
        const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {

        // dt: interval between adjacent updates by seconds.
        double dt    = std::chrono::duration<double>(timestamp - last_update_).count();
        last_update_ = timestamp;

        for (auto& [armor_id, tracker_array] : tracker_map_) {
            for (auto iter = tracker_array.begin(); iter != tracker_array.end();) {
                auto& tracker = *iter;
                if (timestamp - tracker.last_update > std::chrono::milliseconds(predict_duration_))
                    iter = tracker_array.erase(iter);
                else {
                    tracker.Predict(dt);
                    ++iter;
                }
            }
        }

        for (const auto& armor : armors) {
            auto iter = tracker_map_.find(armor.id);
            if (iter != tracker_map_.end()) {
                auto& tracker_array = iter->second;

                if (tracker_array.empty()) {
                    tracker_array.emplace_back(armor, timestamp);
                } else {
                    auto& tracker = tracker_array[0];
                    tracker.Update(armor, timestamp);
                }
            }
        }

        TrackerUnit* selected_tracker = nullptr;
        int selected_level            = 0;
        double minimum_angle          = INFINITY;
        for (auto& [armor_id, tracker_array] : tracker_map_) {
            for (auto& tracker : tracker_array) {
                int level = 0;
                if (tracker.tracking_density > 100)
                    level = 2;
                else if (tracker.tracking_density > 40)
                    level = 1;

                auto center = *fast_tf::cast<rmcs_description::MuzzleLink>(
                    rmcs_description::OdomImu::Position{
                        tracker.ekf.x_(0), tracker.ekf.x_(2), tracker.ekf.x_(4)},
                    tf_);
                double angle = std::acos(center.dot(Eigen::Vector3d{1, 0, 0}) / center.norm());

                if (angle < minimum_angle) {
                    minimum_angle    = angle;
                    selected_tracker = &tracker;
                    selected_level   = level;
                } else if (
                    !std::isinf(angle) && fabs(angle - minimum_angle) < 1e-3
                    && selected_level < level) {
                    selected_level = level;
                    minimum_angle  = angle;

                    selected_tracker = &tracker;
                }
            }
        }
        if (selected_tracker) {
            return std::make_unique<Target>(*selected_tracker);
        } else
            return nullptr;
    }

private:
    struct TrackerUnit {
        TrackerUnit(
            const ArmorPlate3d& armor, const std::chrono::steady_clock::time_point& timestamp)
            : last_update(timestamp) {
            double& r        = r_list[0];
            double yaw       = GetArmorYaw(armor);
            double xc        = armor.position->x() + r * cos(yaw);
            double yc        = armor.position->y() + r * sin(yaw);
            const double& za = armor.position->z();
            // xc  v_xc  yc  v_yc  za  v_za  yaw  v_yaw  r
            ekf.x_ << xc, 0, yc, 0, za, 0, yaw, 0, r;
        }

        void Predict(double dt) {
            ekf.Predict(dt);
            tracked_duration += dt;
            for (bool& updated : armor_newly_updated)
                updated = false;
        }

        void Update(
            const ArmorPlate3d& armor, const std::chrono::steady_clock::time_point& timestamp) {
            double &v_za = ekf.x_(5), &model_yaw = ekf.x_(6), &r = ekf.x_(8);
            double yaw = GetArmorYaw(armor);

            static const double legal_range = Pi / 4;
            static const double step        = 2 * Pi / armor_count;

            double shift = 0;

            size_t i;
            for (i = 0; i < armor_count; ++i) {
                // std::cout << yaw << ' ';
                double diff = GetMinimumAngleDiff(yaw, model_yaw + shift);
                if (-legal_range < diff && diff < legal_range) {
                    yaw = model_yaw + shift + diff;
                    break;
                } else
                    shift += step;
            }

            if (i < 4) {
                model_yaw += shift;
                r = r_list[i];

                Eigen::Vector4d measurement = {
                    armor.position->x(), armor.position->y(), armor.position->z(), yaw};
                ekf.Update(measurement);
                last_update = timestamp;

                v_za = 0;
                if (r > -0.12)
                    r = -0.12;
                else if (r < -0.4)
                    r = -0.4;
                if (i == 0 || i == 2)
                    r_list[0] = r_list[2] = r;
                else if (i == 1 || i == 3)
                    r_list[1] = r_list[3] = r;
                model_yaw -= shift;

                tracked_times += 1;
                if (tracked_duration > 0.5)
                    tracking_density = tracked_times / tracked_duration;

                armor_newly_updated[i] = true;
            }
        }

        [[nodiscard]] static std::tuple<Eigen::Vector3d, double>
            GetArmorState(const Eigen::VectorXd& x, size_t index) {
            const double &xc = x(0), &yc = x(2), &za = x(4), &r = x(8);
            const double yaw_step = 2 * Pi / armor_count;
            double yaw            = x(6) + yaw_step * (double)index;

            double xa = xc - r * cos(yaw);
            double ya = yc - r * sin(yaw);

            return {Eigen::Vector3d(xa, ya, za), yaw};
        }

        [[nodiscard]] std::tuple<Eigen::Vector3d, double> GetArmorState(size_t index) const {
            Eigen::VectorXd x = ekf.x_;
            x(8)              = r_list[index];
            return GetArmorState(x, index);
        }

        EKF ekf;
        static constexpr size_t armor_count = 4;
        std::chrono::steady_clock::time_point last_update;
        double r_list[armor_count]{-0.26, -0.26, -0.26, -0.26};
        bool armor_newly_updated[armor_count]{false, false, false, false};

        double tracked_duration = 0, tracked_times = 0, tracking_density = 0;
    };

    class Target : public TargetInterface {
    public:
        explicit Target(const TrackerUnit& tracker)
            : tracker_(tracker) {}

        [[nodiscard]] rmcs_description::OdomImu::Position Predict(double sec) const override {
            // xc  v_xc  yc  v_yc  za  v_za  yaw  v_yaw  r
            Eigen::VectorXd x = tracker_.ekf.PredictConst(sec);
            const double &xc = x(0), &yc = x(2), &za = x(4), &v_yaw = x(7);
            double& model_yaw = x(6);
            double camera_yaw = std::atan2(-yc, -xc);
            if (fabs(v_yaw) < 12.0) {
                double shift                    = 0;
                static const double legal_range = Pi / 4;
                static const double step        = 2 * Pi / TrackerUnit::armor_count;
                size_t i;
                for (i = 0; i < TrackerUnit::armor_count; ++i) {
                    double diff = GetMinimumAngleDiff(camera_yaw, model_yaw + shift);
                    if (-legal_range < diff && diff < legal_range)
                        break;
                    else
                        shift += step;
                }
                if (i < TrackerUnit::armor_count) {
                    double r   = tracker_.r_list[i];
                    double yaw = model_yaw + shift;
                    auto pos   = Eigen::Vector3d{xc - r * cos(yaw), yc - r * sin(yaw), za};
                    return rmcs_description::OdomImu::Position(pos);
                }
            }
            // return GimbalGyro::Position(0, 0, 0);
            model_yaw = camera_yaw;
            double r  = (tracker_.r_list[0] + tracker_.r_list[1]) / 2;
            // auto [pos, yaw] = TrackerUnit::GetArmorState(x, 0);
            auto pos = Eigen::Vector3d{xc - r * cos(camera_yaw), yc - r * sin(camera_yaw), za};

            return rmcs_description::OdomImu::Position(pos);
        }

    private:
        const TrackerUnit& tracker_;
    };

    int64_t predict_duration_;
    static inline const double Pi = acos(-1);

    static double GetArmorYaw(const ArmorPlate3d& armor) {
        Eigen::Vector3d normal = (*armor.rotation) * Eigen::Vector3d{1, 0, 0};
        return atan2(normal.y(), normal.x());
    }

    static double GetMinimumAngleDiff(double a, double b) {
        double diff = std::fmod(a - b, 2 * Pi);
        if (diff < Pi)
            diff += 2 * Pi;
        else if (diff > Pi)
            diff -= 2 * Pi;
        return diff;
    }

    // Generate continuous yaw (-pi~pi -> -inf~inf)
    static double GetContinuousYaw(const ArmorPlate3d& armor, double last_yaw) {
        double yaw  = GetArmorYaw(armor);
        double diff = GetMinimumAngleDiff(yaw, last_yaw);
        return last_yaw + diff;
    }

    // std::list<TrackerUnit> tracker_array_;
    std::map<ArmorID, std::vector<TrackerUnit>> tracker_map_;
    std::chrono::steady_clock::time_point last_update_;
    const rmcs_description::Tf& tf_;
};

ArmorTracker::ArmorTracker(int64_t predict_duration, const rmcs_description::Tf& tf)
    : pImpl_(new Impl{predict_duration, tf}) {}

std::unique_ptr<TargetInterface> ArmorTracker::Update(
    const std::vector<ArmorPlate3d>& armors, std::chrono::steady_clock::time_point timestamp) {
    return pImpl_->Update(armors, timestamp);
}

ArmorTracker::~ArmorTracker() = default;
