#pragma once

#include <filesystem>
#include "exceptions.h"
#include <functional>

namespace opencv_pipeline {

typedef
enum { noverify=false, verify=true }
verify_result;

struct waitkey
{
    explicit waitkey(int delay = 0) : delay_(delay)
    {
    }

    cv::Mat const &operator()(cv::Mat const &src) const
    {
        cvWaitKey(delay_);
        return src;
    }

  private:
    int const delay_;
};

// image manipulation
cv::Mat gray(cv::Mat const &image);         // single channel grey-scale image
cv::Mat gray_bgr(cv::Mat const &image);     // 3-channel grey-scale image
cv::Mat mirror(cv::Mat const &image);

// early pipeline termination
typedef
enum { end }
pipeline_terminator;

// never required, but here for completeness
inline
cv::Mat operator|(cv::Mat const &image, pipeline_terminator)
{
    return image;
}

struct persistent_pipeline
{
    persistent_pipeline() {}
    explicit persistent_pipeline(std::function<cv::Mat (cv::Mat const &)> &&fn);
    persistent_pipeline &append(std::function<cv::Mat (cv::Mat const &)> &&fn);
    cv::Mat operator()(cv::Mat &&image) const;

  private:
    std::vector<std::function<cv::Mat (cv::Mat const &)>> fn_;
};

}   // namespace opencv_pipeline

#include "detail.h"
#include "opencv_pipeline_impl.inl"
#include "persistent_pipeline.inl"
#include "detail.inl"
