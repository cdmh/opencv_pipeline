// this file is a part of the opencv_pipeline project and contains
// no user-code functions. Don't try to use these functions directly
// from your code. Backward compatibility is not guaranteed.

#pragma once

namespace cv_pipeline {

namespace detail {

std::vector<cv::KeyPoint>
to_keypoints(std::vector<std::vector<cv::Point>> const &regions);

cv::Mat detect_keypoints(
    char const *        const  detector_class,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image);

cv::Mat detect_regions(
    std::string                   const &detector_class,
    std::vector<std::vector<cv::Point>> &regions,
    cv::Mat                              image);

cv::Mat extract_keypoints(
    char const *        const  extractor_class,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image);

cv::Mat extract_regions(
    char const *                  const  extractor_class,
    std::vector<std::vector<cv::Point>> &regions,
    cv::Mat                       const &image);

cv::Mat save(char const * const pathname, cv::Mat const &image);

}   // namespace detail

}   // namespace cv_pipeline
