// this file is a part of the opencv_pipeline project and contains
// no user-code functions. Don't try to use these functions directly
// from your code. Backward compatibility is not guaranteed.

#pragma once

namespace cv_pipeline {

namespace detail {

inline
std::vector<cv::KeyPoint>
to_keypoints(std::vector<std::vector<cv::Point>> const &regions)
{
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(regions.size());
    for (auto const &region : regions)
    {
        auto const rect = fitEllipse(cv::Mat(region));
        auto const d    = rect.size.height * rect.size.width;
        keypoints.emplace_back(rect.center, sqrt(d)/2.0f, rect.angle);
    }
    assert(keypoints.size() == regions.size());
    return keypoints;
}



inline
cv::Mat detect_keypoints(
    char const *        const  detector_class,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image)
{
    auto detector = cv::FeatureDetector::create(detector_class);
    detector->detect(image, keypoints, cv::Mat());
    return image;
}

inline
cv::Mat detect_regions(
    std::string                   const &detector_class,
    std::vector<std::vector<cv::Point>> &regions,
    cv::Mat                              image)
{
    // default values
    int     delta            = 5;
    int     min_area         = 60;
    int     max_area         = 14400;
    double  max_variation    = 0.25;
    double  min_diversity    = 0.2;
    int     max_evolution    = 200;
    double  area_threshold   = 1.01;
    double  min_margin       = 0.003;
    int     edge_blur_size   = 5;
    cv::Mat mask;

    if (detector_class == "MSER")
        image = image | gray;
    else if (detector_class != "MSCR")
        throw exceptions::bad_image();

    cv::MSER mser(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
    mser(image, regions, mask);
    return image;
}

inline
cv::Mat extract_keypoints(
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
cv::Mat extract_regions(
    char const *                  const  extractor_class,
    std::vector<std::vector<cv::Point>> &regions,
    cv::Mat                       const &image)
{
    return extract_keypoints(
        extractor_class,
        to_keypoints(regions),
        image);
}

inline
cv::Mat save(char const * const pathname, cv::Mat const &image)
{
    return cv::imwrite(pathname, image)? image : cv::Mat();
}

}   // namespace detail

}   // namespace cv_pipeline
