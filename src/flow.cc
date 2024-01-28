#include "flow.h"
#include <Eigen/Dense>
#include <iostream>

namespace flow {

cv::Mat
GetFlow(const cv::Mat& prev, const cv::Mat& next)
{
  // Define the return optical flow vector field.
  cv::Mat flow = cv::Mat::zeros(next.size(), CV_32FC2);

  // Convert the images to grayscale.
  cv::Mat prev_gray, next_gray;

  cv::cvtColor(prev, prev_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(next, next_gray, cv::COLOR_BGR2GRAY);

  // Compute Ix and Iy using Sobel operator.
  cv::Mat Ix, Iy;
  int sobel_kernel_size = 5; // Kernel size for Sobel operator
  cv::Sobel(prev_gray, Ix, CV_32F, 1, 0, sobel_kernel_size);
  cv::Sobel(prev_gray, Iy, CV_32F, 0, 1, sobel_kernel_size);
  Ix.convertTo(Ix, CV_32FC1);
  Iy.convertTo(Iy, CV_32FC1);

  // Compute It.
  cv::Mat temporal_diff = next_gray - prev_gray;
  cv::Mat_<float> It;
  temporal_diff.convertTo(It, CV_32FC1);

  cv::GaussianBlur(It, It, cv::Size(5, 5), 0.0);
  cv::GaussianBlur(Ix, Ix, cv::Size(5, 5), 0.0);
  cv::GaussianBlur(Iy, Iy, cv::Size(5, 5), 0.0);

  // For every pixel in the image, compute the optical flow vector for that pixel.
  int patch_size = 3;
  int half_patch_size = patch_size / 2;

  cv::Mat IxIx = Ix.mul(Ix);
  cv::Mat IyIy = Iy.mul(Iy);
  cv::Mat IxIy = Ix.mul(Iy);
  cv::Mat IxIt = Ix.mul(It);
  cv::Mat IyIt = Iy.mul(It);

  cv::boxFilter(IxIx, IxIx, -1, cv::Size(patch_size, patch_size));
  cv::boxFilter(IyIy, IyIy, -1, cv::Size(patch_size, patch_size));
  cv::boxFilter(IxIy, IxIy, -1, cv::Size(patch_size, patch_size));
  cv::boxFilter(IxIt, IxIt, -1, cv::Size(patch_size, patch_size));
  cv::boxFilter(IyIt, IyIt, -1, cv::Size(patch_size, patch_size));

  Eigen::MatrixXf A(2, 2);
  Eigen::VectorXf b(2);

  // Compute the flow for the pixel at (x, y).
  float condition_number_threshold = 3.0f;
  for (int y = 0; y < flow.rows; ++y) {
    for (int x = 0; x < flow.cols; ++x) {
      A(0, 0) = IxIx.at<float>(y, x);
      A(0, 1) = IxIy.at<float>(y, x);
      A(1, 0) = IxIy.at<float>(y, x);
      A(1, 1) = IyIy.at<float>(y, x);
      b(0) = -IxIt.at<float>(y, x);
      b(1) = -IyIt.at<float>(y, x);
      Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      float condition_number =
        svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
      if (svd.singularValues()(svd.singularValues().size() - 1) < 1e-3f) {
        continue;
      }
      if (condition_number > condition_number_threshold or std::isnan(condition_number)) {
        continue;
      }
      Eigen::Vector2f uv = A.inverse() * b;
      if (std::isnan(uv(0)) or std::isnan(uv(1))) {
        continue;
      }
      flow.at<cv::Vec2f>(y, x) = cv::Vec2f(uv(0), uv(1));
    }
  }
  return flow;
}

cv::Mat
GetFlowUsingPyramid(const cv::Mat& image1, const cv::Mat& image2, int num_levels)
{
  cv::Mat flow;
  cv::Mat image1_prev_level;

  for (int level = num_levels - 1; level >= 0; --level) {
    // Compute the current level size.
    cv::Size curr_size = cv::Size(image1.cols / (1 << level), image1.rows / (1 << level));

    // Compute the image 2 for the current level.
    cv::Mat image2_curr_level;
    cv::resize(image2, image2_curr_level, curr_size);

    if (level == num_levels - 1) {
      // If this is the last level, initialize the flow to zero.
      flow = cv::Mat::zeros(curr_size, CV_32FC2);
    } else {
      assert(!flow.empty());
      // Upsample the flow from the previous level to the current level.
      cv::resize(flow, flow, curr_size);
    }

    // Upsample the image 1 using the flow from the previous level.
    image1_prev_level = cv::Mat(curr_size, image1.type());
    cv::Mat image1_curr_level;
    cv::resize(image1_prev_level, image1_curr_level, curr_size);

    // Warp the image 1 using the flow from the previous level.
    cv::Mat warped_image1_curr_level;
    if (level == num_levels - 1) {
      warped_image1_curr_level = image1_curr_level;
    } else {
      cv::Mat map = cv::Mat::zeros(curr_size, CV_32FC2);
      for (int y = 0; y < map.rows; ++y) {
        for (int x = 0; x < map.cols; ++x) {
          cv::Vec2f uv = flow.at<cv::Vec2f>(y, x);
          map.at<cv::Vec2f>(y, x) = cv::Vec2f(x + uv(0), y + uv(1));
        }
      }
      cv::remap(image1_curr_level, warped_image1_curr_level, map, cv::Mat(), cv::INTER_LINEAR);
    }

    // Add the new flow to the flow from the previous level.
    flow += GetFlow(warped_image1_curr_level, image2_curr_level);
  }
  return flow;
};

} // namespace flow