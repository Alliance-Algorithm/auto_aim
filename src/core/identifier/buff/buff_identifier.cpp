#include <openvino/openvino.hpp>

#include "buff_identifier.hpp"

using namespace auto_aim;

class BuffIdentifier::Impl {
public:
    explicit Impl(const std::string& model_path) {
        ov::Core core;
        compiled_model_ = core.compile_model(model_path, "AUTO");
        if (!compiled_model_) {
            throw std::runtime_error("cannot get compiled model.");
        }
        infer_request_ = compiled_model_.create_infer_request();
    }

    std::optional<BuffPlate> Identify(const cv::Mat& img) {
        cv::Mat letterbox_img        = letterbox(img);
        constexpr int letterbox_size = 384; // 352
        double scale                 = letterbox_img.size[0] / (double)letterbox_size;
        cv::Mat blob                 = cv::dnn::blobFromImage(
            letterbox_img, 1.0 / 255.0, cv::Size(letterbox_size, letterbox_size), cv::Scalar(),
            true);
        auto input_port = compiled_model_.input();
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));

        infer_request_.set_input_tensor(input_tensor);

        infer_request_.infer();

        auto output       = infer_request_.get_output_tensor(0);
        auto output_shape = output.get_shape();

        auto* data = output.data<float>();
        cv::Mat output_buffer((int)output_shape[1], (int)output_shape[2], CV_32F, data);
        transpose(output_buffer, output_buffer);

        float confidence[3]     = {.0f, .0f, .0f};
        int confidence_index[3] = {0, 0, 0};

        for (int i = 0; i < output_buffer.rows; ++i) {
            float conf[3];
            for (int j = 4; j < 7; ++j) {
                conf[j - 4] = output_buffer.at<float>(i, j);
            }
            int category = findMaxIndex(conf, 3);
            if (conf[category] > confidence[category]) {
                confidence[category]       = conf[category];
                confidence_index[category] = i;
            }
        }

        if (confidence[1] > 0.25) {
            cv::Mat ToShoot = output_buffer.row(confidence_index[1]);
            BuffPlate result;
            auto get_result_point = [&ToShoot, &scale](int i) {
                auto x = static_cast<float>(ToShoot.at<float>(0, 7 + i * 3) * scale);
                auto y = static_cast<float>(ToShoot.at<float>(0, 7 + i * 3 + 1) * scale);
                return cv::Point2f(x, y);
            };

            result.points.emplace_back(get_result_point(2));
            result.points.emplace_back(get_result_point(1));
            result.points.emplace_back(get_result_point(0));
            result.points.emplace_back(get_result_point(4));
            return result;
        } else
            return {};
    }

private:
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 255), cv::Scalar(170, 0, 255),
        cv::Scalar(255, 0, 85), cv::Scalar(255, 0, 170)};

    [[nodiscard]] static cv::Mat letterbox(const cv::Mat& source) {
        int col        = source.cols;
        int row        = source.rows;
        int _max       = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));
        return result;
    }

    [[nodiscard]] static int findMaxIndex(const float arr[], int size) {
        float maxVal = arr[0];
        int maxIndex = 0;

        for (int i = 1; i < size; i++) {
            if (arr[i] > maxVal) {
                maxVal   = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    [[nodiscard]] static float findMaxValue(const float arr[], int size) {
        float maxVal = arr[0];

        for (int i = 1; i < size; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
            }
        }

        return maxVal;
    }
};

BuffIdentifier::BuffIdentifier(const std::string& model_path)
    : pImpl_(new Impl{model_path}) {}

std::optional<BuffPlate> BuffIdentifier::Identify(const cv::Mat& img) {
    return pImpl_->Identify(img);
}

BuffIdentifier::~BuffIdentifier() = default;