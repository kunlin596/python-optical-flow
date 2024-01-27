#pragma once

#include <opencv2/opencv.hpp>

namespace flow {
cv::Mat
GetFlow(const cv::Mat& prev, const cv::Mat& next);
} // namespace flow
