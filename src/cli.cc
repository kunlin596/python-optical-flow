#include "flow.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Vec3b
AngleToRGB(float angle)
{
  cv::Vec3b color;
  // Normalize angle to [0, 1]
  float normalized = angle / (2.0f * M_PI);

  // Convert to RGB using a simple rainbow mapping
  color[0] = static_cast<uchar>(std::sin(normalized * 2.0f * M_PI) * 127.5f + 127.5f); // Red
  color[1] = static_cast<uchar>(std::sin(normalized * 2.0f * M_PI + 2.0f * M_PI / 3.0f) * 127.5f +
                                127.5f); // Green
  color[2] = static_cast<uchar>(std::sin(normalized * 2.0f * M_PI + 4.0f * M_PI / 3.0f) * 127.5f +
                                127.5f); // Blue

  return color;
}

void
VisualizeOpticalFlow(const cv::Mat& flow, cv::Mat& flowVis)
{
  // Calculate the maximum magnitude
  double maxMag = 0.0;
  for (int y = 0; y < flow.rows; ++y) {
    for (int x = 0; x < flow.cols; ++x) {
      cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
      double mag = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
      if (mag > maxMag) {
        maxMag = mag;
      }
    }
  }

  cv::Mat hsv(flow.rows, flow.cols, CV_8UC3, cv::Scalar(0, 255, 255));
  for (int y = 0; y < flow.rows; ++y) {
    for (int x = 0; x < flow.cols; ++x) {
      cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
      float magnitude = std::sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
      float angle = std::atan2(fxy.y, fxy.x);

      // Normalize magnitude and map to value
      unsigned char value = static_cast<unsigned char>((magnitude / maxMag) * 255.0);
      unsigned char hue = static_cast<unsigned char>((angle + M_PI) * 90 / M_PI);

      hsv.at<cv::Vec3b>(y, x) = cv::Vec3b(hue, 255, value);
    }
  }

  cv::cvtColor(hsv, flowVis, cv::COLOR_HSV2BGR);
}

void
DrawOpticalFlowArrows(cv::Mat& img,
                      const cv::Mat& flow,
                      int step,
                      const cv::Scalar& color,
                      int thickness = 1,
                      int lineType = 8,
                      int shift = 0)
{
  // Draw arrows for a subset of pixels to avoid clutter
  for (int y = 0; y < img.rows; y += step) {
    for (int x = 0; x < img.cols; x += step) {
      // Get the flow vector at this position
      const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);

      // Draw an arrow from (x, y) to (x + fxy.x, y + fxy.y)
      cv::arrowedLine(img,
                      cv::Point(x, y),
                      cv::Point(x + fxy.x * 1000.0f, y + fxy.y * 1000.0f),
                      color,
                      thickness,
                      lineType,
                      shift);
    }
  }
}

int
main(int argc, char** argv)
{
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
    "video-filepath,v", po::value<std::string>(), "set video filepath");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  cv::VideoCapture cap(vm["video-filepath"].as<std::string>());

  cv::Mat frame1;
  cv::Mat frame2;

  cap >> frame1;
  cap >> frame2;

  while (true) {
    if (frame2.empty())
      break;

    float scale = 1.0f;
    cv::Size new_size = cv::Size(int(frame1.cols * scale), int(frame1.rows * scale));
    cv::Mat resized_frame1, resized_frame2;
    cv::resize(frame1, resized_frame1, new_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(frame2, resized_frame2, new_size, 0, 0, cv::INTER_LINEAR);

    cv::Mat flow = flow::GetFlow(resized_frame1, resized_frame2);
    // cv::Mat flow_vis(flow.size(), CV_8UC3);

    // for (int row = 0; row < flow.rows; ++row) {
    //   for (int col = 0; col < flow.cols; ++col) {
    //     cv::Vec2f flow_vec = flow.at<cv::Vec2f>(row, col);
    //     flow_vis.at<cv::Vec3b>(row, col) = AngleToRGB(std::atan2(flow_vec[1], flow_vec[0]));
    //   }
    // }

    // cv::Mat dst = resized_frame2.clone();

    // int step = 2;
    // for (int row = 0; row < flow.rows; ++row) {
    //   for (int col = 0; col < flow.cols; ++col) {
    //     if (row % step != 0 || col % step != 0)
    //       continue;

    //     cv::Vec2f flow_vec = flow.at<cv::Vec2f>(row, col);
    //     float norm = std::sqrt(flow_vec.dot(flow_vec));
    //     if (norm < 1e-3) {
    //       continue;
    //     }
    //     flow_vec /= norm;
    //     flow_vec *= 10.0f;
    //     cv::Point p1(col, row);
    //     cv::Point p2(col + flow_vec[0], row + flow_vec[1]);
    //     cv::arrowedLine(dst, p1, p2, cv::Scalar(0, 0, 255), 1);
    //   }
    // }

    cv::Mat dst = resized_frame1.clone();
    // VisualizeOpticalFlow(flow, dst);
    DrawOpticalFlowArrows(dst, flow, 10, cv::Scalar(0, 0, 255), 1);
    cv::imshow("flow", dst);

    frame1 = frame2.clone();
    cap >> frame2;

    cv::waitKey(1);
  }
}