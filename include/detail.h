// this file is a part of the opencv_pipeline project and contains
// no user-code functions. Don't try to use these functions directly
// from your code. Backward compatibility is not guaranteed.

#pragma once

namespace opencv_pipeline {

namespace detail {

// manuipulation
cv::Mat color_space(cv::Mat const &image, int code);
cv::Mat convert(cv::Mat const &image, int type, double alpha=1.0, double beta=0.0);
cv::Mat dilate(cv::Mat const &image, int dx, int dy);
cv::Mat erode(cv::Mat const &image, int dx, int dy);
cv::Mat gaussian_blur(cv::Mat const &image, int dx, int dy, double sigmaX, double sigmaY, int border);
cv::Mat sobel(cv::Mat const &image, int dx, int dy, int ksize, double scale, double delta, int border);
cv::Mat subtract(cv::Mat const &image1, cv::Mat const &image2);
cv::Mat threshold(cv::Mat const &image, double thresh, double maxval, int type);

// conditions
cv::Mat if_(cv::Mat const &image, std::function<bool const (cv::Mat const &)> cond, pipeline_fn_t fn);

// attributes
bool const channels(cv::Mat const &image, int num);

std::vector<cv::KeyPoint>
to_keypoints(std::vector<std::vector<cv::Point>> const &regions);

cv::Mat detect_keypoints(
    std::string         const &detector,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image);

cv::Mat detect_regions(
    std::string                   const &detector_class,
    std::vector<std::vector<cv::Point>> &regions,
    cv::Mat                              image);

cv::Mat extract_keypoints(
    std::string                      extractor_class,
    std::vector<cv::KeyPoint> const &keypoints,
    cv::Mat                   const &image);

cv::Mat extract_regions(
    std::string                                extractor_class,
    std::vector<std::vector<cv::Point>> const &regions,
    cv::Mat                             const &image);

cv::Mat        save(cv::Mat const &image, std::filesystem::path pathname);
cv::Mat const &show(char const * const window_name, cv::Mat const &image);

}   // namespace detail

}   // namespace opencv_pipeline
