// this file is a part of the opencv_pipeline project and contains
// no user-code functions. Don't try to use these functions directly
// from your code. Backward compatibility is not guaranteed.

#pragma once

namespace cv_pipeline {

namespace detail {

inline
cv::Mat detect(
    char const *        const  detector_class,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image)
{
    auto detector = cv::FeatureDetector::create(detector_class);
    detector->detect(image, keypoints, cv::Mat());
    return image;
}

inline
cv::Mat extract(
    char const *        const  extractor_class,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image)
{
    auto extractor = cv::DescriptorExtractor::create(extractor_class);

    cv::Mat descriptors;
    extractor->compute(image, keypoints, descriptors);
    return descriptors;
}

inline
cv::Mat save(char const * const pathname, cv::Mat const &image)
{
    return cv::imwrite(pathname, image)? image : cv::Mat();
}

}   // namespace detail

}   // namespace cv_pipeline
