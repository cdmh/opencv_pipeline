#pragma once

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

// image loading
cv::Mat load(char const * const pathname);
cv::Mat load(std::string const &pathname);

// save an image
std::function<cv::Mat (cv::Mat const &)>
save(char const * const pathname);

// image manipulation
cv::Mat gray(cv::Mat const &image);         // single channel grey-scale image
cv::Mat gray_bgr(cv::Mat const &image);     // 3-channel grey-scale image
cv::Mat mirror(cv::Mat const &image);

// early pipeline termination
struct ending {};
ending end() { return ending(); }

// never required, but here for completeness
inline
cv::Mat operator|(cv::Mat const &img, std::function<ending ()>)
{
    return img;
}

}   // namespace opencv_pipeline

#include "detail.h"
#include "opencv_pipeline_impl.inl"
#include "persistent_pipeline.inl"
#include "detail.inl"
