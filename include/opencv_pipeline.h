#pragma once

#include "exceptions.h"
#include <functional>

namespace cv_pipeline {

// load an image
inline
cv::Mat load(char const * const pathname)
{
    return cv::imread(pathname);
}

// load an image
inline
cv::Mat load(std::string const &pathname)
{
    return cv::imread(pathname);
}

// save an image
std::function<cv::Mat (cv::Mat const &)>
save(char const * const pathname);

// detect keypoint features
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<cv::KeyPoint> &keypoints);

// detect region features
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<std::vector<cv::Point>> &regions);

std::function<cv::Mat (cv::Mat const &)>
extract(char const * const detector, std::vector<cv::KeyPoint> &keypoints);

inline
cv::Mat grey(cv::Mat image)
{
    cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cvtColor(image, image, cv::COLOR_GRAY2BGR);
    return image;
}

inline
cv::Mat mirror(cv::Mat image)
{
    flip(image, image, 1);
    return image;
}

// apply a function to an image
inline
cv::Mat operator|(cv::Mat const &left, cv::Mat(*right)(cv::Mat const &))
{
    return right(left);
}

// apply a function to an image - enables std::bind() bound parameters
inline
cv::Mat operator|(cv::Mat const &left, std::function<cv::Mat(cv::Mat const &)> right)
{
    return right(left);
}

typedef enum { noverify=false, verify=true } verify_result;

// load an image with optional verification.
// whether or not to verify is compulsory -- verify or noverify must
// go between a load and any subsequent manipulations through the
// pipeline interface
inline
cv::Mat operator|(char const * const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::image_not_found(pathname);
    return image;
}

inline
cv::Mat operator|(std::string const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::image_not_found(pathname.c_str());
    return image;
}

// verify an image is not empty
inline
cv::Mat operator|(cv::Mat const &image, verify_result verify)
{
    if (verify  &&  image.empty())
        throw exceptions::bad_image();
    return image;
}

}   // namespace cv_pipeline

#include "detail.inl"


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
