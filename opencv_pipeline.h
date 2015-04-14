#pragma once

#include "exceptions.h"
#include <functional>

namespace cv_pipeline {

cv::Mat grey(cv::Mat const &img)
{
    cv::Mat image;
    cvtColor(img, image, cv::COLOR_BGR2GRAY);
    cvtColor(image, image, cv::COLOR_GRAY2BGR);
    return image;
}

cv::Mat load(char const * const pathname)
{
    return cv::imread(pathname);
}

cv::Mat load(std::string const &pathname)
{
    return cv::imread(pathname);
}

cv::Mat mirror(cv::Mat const &img)
{
    cv::Mat image;
    cv::flip(img, image, 1);
    return image;
}

cv::Mat operator|(cv::Mat const &left, cv::Mat(*right)(cv::Mat const &))
{
    return right(left);
}

typedef enum { noverify=false, verify=true } verify_result;
cv::Mat operator|(char const * const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::image_not_found(pathname);
    return image;
}

}   // namespace cv_pipeline
