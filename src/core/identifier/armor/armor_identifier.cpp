#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include <rmcs_core/msgs.hpp>

#include "core/identifier/armor/armor.hpp"
#include "core/identifier/armor/color_identifier.hpp"
#include "core/identifier/armor/number_identifier.hpp"

#include "armor_identifier.hpp"

using namespace auto_aim;

class ArmorIdentifier::Impl {
public:
    template <class... Args>
    explicit Impl(Args&&... args)
        : _blueIdentifier(BlueLightBarHue)
        , _redIdentifier(RedLightBarHue)
        , _numberIdentifier(std::forward<Args>(args)...) {}

    std::vector<ArmorPlate>
        Identify(const cv::Mat& img, const rmcs_core::msgs::RoboticColor& target_color) {
        cv::Mat imgThre, imgGray;
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        cv::threshold(imgGray, imgThre, 150, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<LightBar> lightBars;
        std::vector<ArmorPlate> result;
        cv::findContours(imgThre, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (const auto& contour : contours) {
            if (auto&& lightBarOpt = _solveToLightbar(img, contour, target_color)) {
                lightBars.push_back(*lightBarOpt);

                std::sort(lightBars.begin(), lightBars.end(), [](LightBar& a, LightBar& b) {
                    return a.top.x < b.top.x;
                });
            }
        }

        size_t&& lightBarsSize = lightBars.size();
        for (size_t i = 0; i < lightBarsSize; ++i) {
            float Isize         = P2PDis(lightBars[i].top, lightBars[i].bottom);
            cv::Point2f Icenter = (lightBars[i].top + lightBars[i].bottom) / 2;
            for (size_t j = i + 1; j < lightBarsSize; ++j) { // 一些筛选条件

                float Jsize = P2PDis(lightBars[j].top, lightBars[j].bottom);
                if (fmax(Isize, Jsize) / fmin(Isize, Jsize) > maxArmorLightRatio)
                    continue;
                if (fabs(lightBars[i].angle - lightBars[j].angle) > maxdAngle)
                    continue;
                if (malposition(lightBars[i], lightBars[j]) > maxMalposition)
                    continue;
                cv::Point2f Jcenter = (lightBars[j].top + lightBars[j].bottom) / 2;
                if (fabs(Icenter.y - Jcenter.y) * 2 / (Isize + Jsize) > maxLightDy)
                    continue;
                float lightBarDis = P2PDis(Icenter, Jcenter) * 2 / (Isize + Jsize);
                if (lightBarDis > bigArmorDis)
                    continue;

                ArmorPlate armor(lightBars[i], lightBars[j], ArmorID::Unknown, lightBarDis > 3.5);

                if (_numberIdentifier.Identify(imgGray, armor))
                    result.push_back(armor);
            }
        }
        return result;
    }

private:
    ColorIdentifier _blueIdentifier, _redIdentifier;
    NumberIdentifier _numberIdentifier;

    inline static constexpr const double maxArmorLightRatio = 1.5;
    inline static constexpr const double maxdAngle          = 9.5;
    inline static constexpr const double maxMalposition     = 0.7;
    inline static constexpr const double maxLightDy         = 0.9;
    inline static constexpr const double bigArmorDis        = 5.0;
    inline static constexpr const double BlueLightBarHue    = 228.0f;
    inline static constexpr const double RedLightBarHue     = 11.0f;

    static inline double malposition(const LightBar& LBl, const LightBar& LBr) {
        cv::Point2f axis = (LBl.top - LBl.bottom + LBr.top - LBr.bottom) / 2;
        cv::Point2f dis  = (LBl.top + LBl.bottom - LBr.top - LBr.bottom) / 2;
        return fabs(axis.dot(dis) / axis.cross(dis));
    }

    static inline float P2PDis(const cv::Point2f& a, const cv::Point2f& b) {
        cv::Point2f tmp = b - a;
        return sqrt(tmp.x * tmp.x + tmp.y * tmp.y);
    }

    static inline float P2PDis(const cv::Point3f& a, const cv::Point3f& b) {
        cv::Point3f tmp = b - a;
        return sqrt(tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z);
    }

    template <typename T>
    inline void LimitRange(T& value, const T min, const T max) {
        value = value < min ? min : value;
        value = value > max ? max : value;
    }

    std::optional<LightBar> _solveToLightbar(
        const cv::Mat& img, const std::vector<cv::Point>& contour,
        const rmcs_core::msgs::RoboticColor& target_color) {
        auto&& contourSize = contour.size();
        if (contourSize >= 5) {
            float scoreMap[256];
            float confidence                                            = 0.0f;
            scoreMap[static_cast<size_t>(ColorConfidence::NotCredible)] = 0.0f;
            scoreMap[static_cast<size_t>(ColorConfidence::CredibleOneChannelOverexposure)] =
                1.0f / (float)contourSize;
            scoreMap[static_cast<size_t>(ColorConfidence::CredibleTwoChannelOverexposure)] =
                0.5f / (float)contourSize;
            scoreMap[static_cast<size_t>(ColorConfidence::CredibleThreeChannelOverexposure)] =
                0.2f / (float)contourSize;

            auto& colorIdentifier = target_color == rmcs_core::msgs::RoboticColor::Blue
                                      ? _blueIdentifier
                                      : _redIdentifier;
            int maxPointY         = 0;
            for (const auto& point : contour) {
                maxPointY = std::max(maxPointY, point.y);
                auto c    = reinterpret_cast<const uchar*>(&img.at<cv::Vec3b>(point));
                confidence += scoreMap[static_cast<size_t>(colorIdentifier.Identify(c))];
            }
            if (img.rows == maxPointY + 1)
                confidence = 0;

            if (confidence > 0.45f) {
                constexpr int angleRange = 30;
                auto box                 = cv::minAreaRect(contour);

                cv::Point2f corner[4];
                box.points(corner);
                auto diff = box.size.width - box.size.height;
                if (diff > 0) {        // 旋转前，矩形横放
                    float angle = fmodf(box.angle + 360, 360);
                    if (90 - angleRange < angle && angle < 90 + angleRange) {
                        return LightBar(
                            (corner[0] + corner[1]) / 2, (corner[2] + corner[3]) / 2, 0);
                    } else if (270 - angleRange < angle && angle < 270 + angleRange) {
                        return LightBar(
                            (corner[2] + corner[3]) / 2, (corner[0] + corner[1]) / 2, 0);
                    }
                } else if (diff < 0) { // 旋转前，矩形竖放
                    float angle = fmodf(box.angle + 360 + 90, 360);
                    if (90 - angleRange < angle && angle < 90 + angleRange) {
                        return LightBar(
                            (corner[1] + corner[2]) / 2, (corner[0] + corner[3]) / 2, 0);
                    } else if (270 - angleRange < angle && angle < 270 + angleRange) {
                        return LightBar(
                            (corner[0] + corner[3]) / 2, (corner[1] + corner[2]) / 2, 0);
                    }
                }
            }
        }
        return std::nullopt;
    }
};

ArmorIdentifier::ArmorIdentifier(const std::string& model_path)
    : pImpl_(new Impl{model_path}) {}

std::vector<ArmorPlate> ArmorIdentifier::Identify(
    const cv::Mat& img, const rmcs_core::msgs::RoboticColor& target_color) {
    return pImpl_->Identify(img, target_color);
}

ArmorIdentifier::~ArmorIdentifier() = default;