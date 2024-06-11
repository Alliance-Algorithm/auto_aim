/**
 * @file color_identifier.hpp
 * @author Lorenzo Feng (lorenzo.feng@njust.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-06-07
 *
 * (C)Copyright: NJUST.Alliance - All rights reserved
 *
 */
#pragma once

#include <cmath>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

namespace auto_aim {

enum class ColorConfidence : uchar {
    NotCredible                      = 0,
    CredibleZeroChannelOverexposure  = 255,
    CredibleOneChannelOverexposure   = 191,
    CredibleTwoChannelOverexposure   = 127,
    CredibleThreeChannelOverexposure = 63
};

class ColorIdentifier {
public:
    explicit ColorIdentifier(float hue360);

    ColorConfidence Identify(const uchar* color) const;

private:
    // 过曝顺序
    int _cc1, _cc2, _cc3;

    static constexpr float _minSaturation = 0.8, _minValue = 0.5;
    static constexpr uchar _minSaturationUChar = _minSaturation * 255,
                           _minValueUchar      = static_cast<uchar>(_minValue * 255);
    float _minHue, _maxHue;
    uchar _minHueUChar, _maxHueUChar;

    cv::Mat _confidenceMap;

    [[nodiscard]] cv::Mat GenerateMap() const {
        // 暂时只能生成蓝色和红色
        constexpr int renderSize = 256;

        cv::Mat imgBGR, imgHSV, imgMap;
        imgBGR.create(renderSize, renderSize, CV_32FC3);
        imgMap.create(renderSize, renderSize, CV_8UC1);

        for (int i = 0; i < renderSize; ++i) {
            for (int j = 0; j < renderSize; ++j) {
                cv::Vec3f color;
                color[_cc1]                = 1.0f;
                color[_cc2]                = static_cast<float>(i) / renderSize;
                color[_cc3]                = static_cast<float>(j) / renderSize;
                imgBGR.at<cv::Vec3f>(j, i) = color;
                imgMap.at<uchar>(j, i)     = static_cast<uchar>(ColorConfidence::NotCredible);
            }
        }
        cv::cvtColor(imgBGR, imgHSV, cv::COLOR_BGR2HSV);
        int cornerPointX = 0;
        float maxSlope   = 0;
        int maxFitY[renderSize];
        for (int& i : maxFitY)
            i = -1;
        for (int i = 0; i < renderSize; ++i) {     // x
            for (int j = 0; j < renderSize; ++j) { // y
                auto& hsv = imgHSV.at<cv::Vec3f>(j, i);
                if (_minHue < hsv[0] && hsv[0] < _maxHue && hsv[1] > _minSaturation
                    && hsv[2] > _minValue) {
                    imgMap.at<uchar>(j, i) =
                        static_cast<uchar>(ColorConfidence::CredibleZeroChannelOverexposure);
                    maxFitY[i] = j;
                    if (i > 0) {
                        float slope = static_cast<float>(j) / (float)i;
                        if (slope > maxSlope) {
                            maxSlope     = slope;
                            cornerPointX = i;
                        }
                    }
                } else
                    imgBGR.at<cv::Vec3f>(j, i) = {1.0f, 1.0f, 1.0f};
            }
        }
        for (int i = 0; i < renderSize; ++i) {     // x
            for (int j = 0; j < maxFitY[i]; ++j) { // y
                auto& pixel = imgMap.at<uchar>(j, i);
                if (pixel != static_cast<uchar>(ColorConfidence::CredibleZeroChannelOverexposure))
                    pixel = static_cast<uchar>(ColorConfidence::CredibleOneChannelOverexposure);
            }
        }
        for (int i = cornerPointX; i < renderSize; ++i) {
            int maxY = (int)lround(maxSlope * (float)i);
            for (int j = 0; j <= maxY; ++j) {
                auto& pixel = imgMap.at<uchar>(j, i);
                if (pixel != static_cast<uchar>(ColorConfidence::CredibleZeroChannelOverexposure))
                    pixel = static_cast<uchar>(ColorConfidence::CredibleOneChannelOverexposure);
            }
        }
        for (int j = 0; j < renderSize; ++j) {
            if (j == renderSize - 1)
                imgMap.at<uchar>(j, renderSize - 1) =
                    static_cast<uchar>(ColorConfidence::CredibleThreeChannelOverexposure);
            else
                imgMap.at<uchar>(j, renderSize - 1) =
                    static_cast<uchar>(ColorConfidence::CredibleTwoChannelOverexposure);
        }

        if constexpr (renderSize == 256) {
            return imgMap;
        } else {
            auto& imgDisplay = false ? imgBGR : imgMap;
            cv::flip(imgDisplay, imgDisplay, 0);
            cv::imshow("debug", imgDisplay);
            cv::waitKey(-1);
            throw(std::runtime_error("render size != 256, enable debugging..."));
        }
    }
};

} // namespace auto_aim
