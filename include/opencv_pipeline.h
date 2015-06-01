#pragma once

#include "exceptions.h"
#include <functional>

namespace cv_pipeline {

typedef
enum { noverify=false, verify=true }
verify_result;

// image loading
cv::Mat load(char const * const pathname);
cv::Mat load(std::string const &pathname);

// save an image
std::function<cv::Mat (cv::Mat const &)>
save(char const * const pathname);

// detect keypoint features
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<cv::KeyPoint> &keypoints);

// detect region features
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<std::vector<cv::Point>> &regions);

// extract region features
std::function<cv::Mat (cv::Mat const &)>
extract(char const * const detector, std::vector<cv::KeyPoint> &keypoints);

// image manipulation
cv::Mat gray(cv::Mat image);        // single channel grey-scale image
cv::Mat gray_bgr(cv::Mat image);    // 3-channel grey-scale image
cv::Mat mirror(cv::Mat image);

}   // namespace cv_pipeline

#include "detail.h"
#include "opencv_pipeline_impl.inl"
#include "detail.inl"
