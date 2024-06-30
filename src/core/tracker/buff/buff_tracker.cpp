
#include "buff_tracker.hpp"
#include "core/tracker/target.hpp"

#include <ceres/ceres.h>
#include <rmcs_description/tf_description.hpp>

using namespace auto_aim;

class BuffTracker::Impl {
public:
    explicit Impl(int64_t predict_duration)
        : predict_duration_(predict_duration) {}

    std::unique_ptr<TargetInterface>
        Update(const BuffPlate3d& buff, std::chrono::steady_clock::time_point timestamp) {
        if (!angle_array_.empty()
            && timestamp - last_update_ > std::chrono::milliseconds(predict_duration_)) {
            time_array_.clear();
            angle_array_.clear();
        }

        Eigen::Vector3d vec = *buff.rotation * Eigen::Vector3d{1, 0, 0};
        vec                 = Eigen::AngleAxisd(yaw_offset_, Eigen::Vector3d::UnitZ()) * vec;
        if (vec.y() * vec.y() + vec.z() * vec.z() < 0.1)
            return {};
        double angle = atan2(vec.z(), vec.y());

        if (angle_array_.empty()) {
            first_update_ = last_update_ = timestamp;
            angle_sum_                   = 0;
            angle_offset_                = -angle;
            time_array_.push_back(0);
            angle_array_.push_back(0);
        } else {
            const double& last_angle = angle_array_[angle_array_.size() - 1];
            angle    = last_angle + GetMinimumAngleDiff(angle + angle_offset_, last_angle);
            double t = std::chrono::duration<double>(timestamp - first_update_).count();
            time_array_.push_back(t);
            angle_array_.push_back(angle);
            angle_sum_ += angle;
            last_update_ = timestamp;
            if (angle_array_.size() > 10) {
                double k = 1.0, b = 0.0, linear_loss;
                bool linear_succeed = LinearFit(angle_sum_ > 0, k, b, linear_loss);
                double a = 0.9125, omega = 1.942, phi = 0, c = 0, sine_loss;
                bool sine_succeed = SineFit(angle_sum_ > 0, a, omega, phi, c, sine_loss);

                if (linear_succeed && sine_succeed) {
                    if (linear_loss < sine_loss)
                        return std::make_unique<Target>(buff, 0.7, angle_sum_ > 0, k, t);
                    else
                        return std::make_unique<Target>(
                            buff, 0.7, angle_sum_ > 0, a, omega, phi, t);
                }
            }
        }

        return nullptr;
    }

    void ResetAll() {}

private:
    class Target : public TargetInterface {
    public:
        Target(
            const BuffPlate3d& buff, const double& radius, bool is_rotation_forward,
            const double& k, const double& t)
            : position_(*buff.position)
            , is_rotation_forward_(is_rotation_forward) {
            center_line_ = *buff.rotation * Eigen::Vector3d{-radius, 0, 0};
            center_line_ += *buff.position;
            center_line_.normalize();
            k_        = k;
            begin_x_  = t;
            begin_y_  = CalcLinear(t);
            is_linear = true;
        }

        Target(
            const BuffPlate3d& buff, const double& radius, bool is_rotation_forward,
            const double& a, const double& omega, const double& phi, const double& t)
            : position_(*buff.position)
            , is_rotation_forward_(is_rotation_forward) {
            center_line_ = *buff.rotation * Eigen::Vector3d{-radius, 0, 0};
            center_line_ += *buff.position;
            center_line_.normalize();
            a_        = a;
            omega_    = omega;
            phi_      = phi;
            begin_x_  = t;
            begin_y_  = CalcSine(t);
            is_linear = false;
        }

        [[nodiscard]] rmcs_description::OdomImu::Position Predict(double sec) const override {
            double x      = begin_x_ + sec;
            double angle  = (is_linear ? CalcLinear(x) : CalcSine(x)) - begin_y_;
            angle         = is_rotation_forward_ ? angle : -angle;
            auto rotation = Eigen::AngleAxisd{angle, center_line_};
            return rmcs_description::OdomImu::Position{rotation * position_};
        }

    private:
        [[nodiscard]] double CalcLinear(const double& t) const { return k_ * t; }

        [[nodiscard]] double CalcSine(const double& t) const {
            return (2.09 - a_) * t + a_ * (cos(phi_) - cos(phi_ + omega_ * t)) / omega_;
        }

        Eigen::Vector3d position_, center_line_;
        double k_;
        double a_, omega_, phi_;
        double begin_x_, begin_y_;
        bool is_linear, is_rotation_forward_;
    };

    bool SineFit(bool is_forward, double& a, double& omega, double& phi, double& c, double& cost) {
        ceres::Problem problem;
        for (size_t i = 0; i < angle_array_.size(); ++i) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1, 1, 1>(
                    new ExponentialResidual(
                        time_array_[i], is_forward ? angle_array_[i] : -angle_array_[i]));
            problem.AddResidualBlock(cost_function, nullptr, &a, &omega, &phi, &c);
        }
        ceres::Solver::Options options;
        options.max_num_iterations = 20;
        options.linear_solver_type = ceres::DENSE_QR;
        //        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cost = summary.final_cost;
        return summary.IsSolutionUsable();
        //        std::cout << "SineFit: " << summary.BriefReport() << "\n";
        //        std::cout << "Final: " << k << ' ' << b << "\n";
    }

    bool LinearFit(bool is_forward, double& k, double& b, double& cost) {
        ceres::Problem problem;
        for (size_t i = 0; i < angle_array_.size(); ++i) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<ExponentialResidual2, 1, 1, 1>(
                    new ExponentialResidual2(
                        time_array_[i], is_forward ? angle_array_[i] : -angle_array_[i]));
            problem.AddResidualBlock(cost_function, nullptr, &k, &b);
        }
        ceres::Solver::Options options;
        options.max_num_iterations = 20;
        options.linear_solver_type = ceres::DENSE_QR;
        //        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        cost = summary.final_cost;
        return summary.IsSolutionUsable();
        //        std::cout << "LinearFit: " << summary.BriefReport() << "\n";
        //        std::cout << "Final: " << k << ' ' << b << "\n";
    }
    static double GetMinimumAngleDiff(double a, double b) {
        double diff = std::fmod(a - b, 2 * Pi / 5);
        if (diff < -Pi / 5)
            diff += 2 * Pi / 5;
        else if (diff > Pi / 5)
            diff -= 2 * Pi / 5;
        return diff;
    }

    struct ExponentialResidual {
        ExponentialResidual(double x, double y)
            : x_(x)
            , y_(y) {}
        template <typename T>
        bool operator()(
            const T* const a, const T* const omega, const T* const phi, const T* const c,
            T* residual) const {
            auto value =
                (T(2.09) - a[0]) * T(x_)
                + a[0] * (ceres::cos(phi[0]) - ceres::cos(phi[0] + omega[0] * T(x_))) / omega[0]
                + c[0];
            residual[0] = T(y_) - value;
            return true;
        }

    private:
        const double x_;
        const double y_;
    };

    struct ExponentialResidual2 {
        ExponentialResidual2(double x, double y)
            : x_(x)
            , y_(y) {}
        template <typename T>
        bool operator()(const T* const k, const T* const b, T* residual) const {
            auto value  = k[0] * T(x_) + b[0];
            residual[0] = T(y_) - value;
            return true;
        }

    private:
        const double x_;
        const double y_;
    };

    static inline const double Pi = acos(-1);

    double angle_sum_, angle_offset_, yaw_offset_ = 0;
    std::chrono::steady_clock::time_point first_update_, last_update_;
    std::vector<double> time_array_ = {}, angle_array_ = {};
    int64_t predict_duration_;
};

BuffTracker::BuffTracker(int64_t predict_duration)
    : pImpl_(new Impl{predict_duration}) {}

std::unique_ptr<TargetInterface>
    BuffTracker::Update(const BuffPlate3d& buff, std::chrono::steady_clock::time_point timestamp) {
    return pImpl_->Update(buff, timestamp);
}

void BuffTracker::ResetAll() { return pImpl_->ResetAll(); }

BuffTracker::~BuffTracker() = default;