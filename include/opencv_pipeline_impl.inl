#pragma once

namespace cv_pipeline {

inline
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<cv::KeyPoint> &keypoints)
{
    return std::bind(detail::detect_keypoints, detector, std::ref(keypoints), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<std::vector<cv::Point>> &regions)
{
    return std::bind(detail::detect_regions, detector, std::ref(regions), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
extract(char const * const detector, std::vector<cv::KeyPoint> &keypoints)
{
    return std::bind(detail::extract_keypoints, detector, std::ref(keypoints), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
extract(char const * const detector, std::vector<std::vector<cv::Point>> &regions)
{
    return std::bind(detail::extract_regions, detector, std::ref(regions), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
save(char const * const pathname)
{
    return std::bind(detail::save, pathname, std::placeholders::_1);
}

}   // namespace cv_pipeline
