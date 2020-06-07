#pragma once

#include <filesystem>
#include "exceptions.h"
#include <functional>
#include <array>
#include <vector>

namespace opencv_pipeline {

using pipeline_fn_t = std::function<cv::Mat (cv::Mat const &)>;

struct waitkey
{
    explicit waitkey(int delay) : delay_(delay)
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

// result verification
typedef
enum { noverify=false, verify=true }
verify_result;

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
    explicit persistent_pipeline(pipeline_fn_t &&fn);
    persistent_pipeline &append(pipeline_fn_t &&fn);
    cv::Mat operator()(cv::Mat &&image) const;

  private:
    std::vector<pipeline_fn_t> fn_;
};

}   // namespace opencv_pipeline

#include "detail.h"
#include "opencv_pipeline_impl.inl"
#include "persistent_pipeline.inl"
#include "detail.inl"
