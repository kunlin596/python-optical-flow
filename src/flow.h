#pragma once

#include <opencv2/opencv.hpp>

namespace flow {
cv::Mat
GetFlow(const cv::Mat& prev, const cv::Mat& next);

cv::Mat
GetFlowUsingPyramid(const cv::Mat& image1, const cv::Mat& image2, int num_levels = 3);
} // namespace flow
