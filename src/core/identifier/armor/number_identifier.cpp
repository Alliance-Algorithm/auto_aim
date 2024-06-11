#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include "number_identifier.hpp"

using namespace auto_aim;

NumberIdentifier::NumberIdentifier(const std::string& model_path) {
    _net = cv::dnn::readNetFromONNX(model_path);
    if (_net.empty()) {
        throw std::runtime_error("Could not read the model file.");
    }
}

bool NumberIdentifier::Identify(const cv::Mat& imgGray, ArmorPlate& armor) {
    static const std::vector<cv::Point2f> dst = {
        {-4.5, 10.125},
        {-4.5, 25.875},
        {40.5, 25.875},
        {40.5, 10.125}
    };
    cv::Mat imgWarped, imgNumber, imgInput, M = getPerspectiveTransform(armor.points, dst);

    warpPerspective(imgGray, imgWarped, M, cv::Size(36, 36));

    double imgWrapedMaxVal;
    minMaxLoc(imgWarped, nullptr, &imgWrapedMaxVal, nullptr, nullptr);
    if (imgWrapedMaxVal > 120)
        return false;

    threshold(imgWarped, imgNumber, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat blobImage = cv::dnn::blobFromImage(imgNumber, 1.0, cv::Size(36, 36), false, false);

    _net.setInput(blobImage);
    cv::Mat pred = _net.forward();

    double maxVal;
    cv::Point maxLoc;
    minMaxLoc(pred, nullptr, &maxVal, nullptr, &maxLoc);

    switch (maxLoc.x) {
    case 0: return false;
    case 1:
        armor.id             = ArmorID::Hero;
        armor.is_large_armor = true;
        break;
    case 2:
        armor.id             = ArmorID::Engineer;
        armor.is_large_armor = false;
        break;
    case 3:
        armor.id             = ArmorID::InfantryIII;
        armor.is_large_armor = false;
        break;
    case 4:
        armor.id             = ArmorID::InfantryIV;
        armor.is_large_armor = false;
        break;
    case 5:
        armor.id             = ArmorID::InfantryV;
        armor.is_large_armor = false;
        break;
    case 6:
        armor.id             = ArmorID::Sentry;
        armor.is_large_armor = false;
        break;
    case 7:
        armor.id             = ArmorID::InfantryIII;
        armor.is_large_armor = true;
        break;
    case 8:
        armor.id             = ArmorID::InfantryIV;
        armor.is_large_armor = true;
        break;
    case 9:
        armor.id             = ArmorID::InfantryV;
        armor.is_large_armor = true;
        break;
    case 10:
        armor.id             = ArmorID::Outpost;
        armor.is_large_armor = false;
        break;
    case 11:
        armor.id             = ArmorID::Base;
        armor.is_large_armor = false;
        break;
    default: return false;
    }

    return true;
}
