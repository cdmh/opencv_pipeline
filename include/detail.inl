// this file is a part of the opencv_pipeline project and contains
// no user-code functions. Don't try to use these functions directly
// from your code. Backward compatibility is not guaranteed.

#pragma once

namespace opencv_pipeline {

namespace detail {

inline
cv::Mat color_space(cv::Mat const &image, int code)
{
    cv::Mat dst;
    cv::cvtColor(image, dst, code);
    return dst;
}

inline
cv::Mat convert(cv::Mat const &image, int type, double alpha, double beta)
{
    if (image.type() == type)
        return image;

    cv::Mat dst;
    image.convertTo(dst, type, alpha, beta);
    return dst;
}

inline
cv::Mat dilate(cv::Mat const &image, int dx, int dy)
{
    cv::Mat dst;
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(dx, dy));
    dilate(image, dst, kernel);
    return dst;
}

inline
cv::Mat erode(cv::Mat const &image, int dx, int dy)
{
    cv::Mat dst;
    cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(dx, dy));
    erode(image, dst, kernel);
    return dst;
}

inline
cv::Mat gaussian_blur(cv::Mat const &image, int dx, int dy, double sigmaX, double sigmaY, int border)
{
    cv::Mat dst;
    if (image.depth() != CV_64F)
        image.convertTo(dst, CV_MAKETYPE(CV_64F,image.channels()));
    else
        dst = image;

    // apply Gaussian Blur using 64-bit floating point to maintain precision
    cv::GaussianBlur(dst, dst, cv::Size(dx, dy), sigmaX, sigmaY, border);

    if (image.depth() != CV_64F)
        dst.convertTo(dst, image.type());
    return dst;
}

inline
cv::Mat sobel(cv::Mat const &image, int dx, int dy, int ksize, double scale, double delta, int border)
{
    cv::Mat dst;
    cv::Sobel(image, dst, image.depth(), dx, dy, ksize, scale, delta, border);
#ifndef NDEBUG
    cv::Mat s1; 
    image.convertTo(s1, CV_8UC1);
    cv::Mat d1; 
    dst.convertTo(d1, CV_8UC1);
#endif
    return dst;
}

inline
cv::Mat subtract(cv::Mat const &image1, cv::Mat const &image2)
{
    return image1 - image2;
}

inline
cv::Mat threshold(cv::Mat const &image, double thresh, double maxval, int type)
{
    cv::Mat dst;
    cv::threshold(image, dst, thresh, maxval, type);
    return dst;
}


//
// conditions
//

inline
cv::Mat if_(
    cv::Mat                              const &image,
    std::function<bool const (cv::Mat const &)> cond,
    std::function<cv::Mat (cv::Mat const &)>    fn)
{
    return cond(image)? fn(image) : image;
}


//
// image attributes
//

inline
bool const channels(cv::Mat const &image, int num)
{
    return image.channels() == num;
}


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
    std::string         const &detector_class,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat             const &image)
{
    auto detector = cv::FeatureDetector::create(detector_class);
    detector->detect(image, keypoints, cv::Mat());
    return image;
}

inline
cv::Mat detect_regions(
    std::string                   const &detector,
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

    if (detector == "MSER")
        image = image | gray;
    else if (detector != "MSCR")
        throw exceptions::bad_image();

    // MSCR is implemented by the MSER detector and automatically
    // used if detect() is given a colour image
    cv::MSER mser(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
    mser(image, regions, mask);
    return image;
}

// copy each descriptor in turn, skipping ones without a keypoint
inline
void copy_keypoint_descriptors(
    cv::Mat                   const &src,
    std::vector<cv::KeyPoint> const &src_kps,
    cv::Mat                         &dst,
    std::vector<cv::KeyPoint> const &dst_kps)
{
    assert(src.type() == dst.type());
    assert(src_kps.size() < dst_kps.size());
    assert(src_kps.size() == (size_t)src.rows);
    assert(dst_kps.size() == (size_t)dst.rows);

    int const elem_size = CV_ELEM_SIZE(src.type());
    int const width     = src.cols * elem_size;
    for (int d=0, s=0; s<int(src_kps.size()); ++d)
    {
        if (src_kps[s].pt == dst_kps[d].pt)
        {
            std::copy_n(src.ptr(s), width, dst.ptr(d));
            ++s;
        }
    }
}

inline
cv::Mat extract_keypoints(
    char const *              const  extractor_class,
    std::vector<cv::KeyPoint> const &keypoints,
    cv::Mat                   const &image)
{
    auto extractor = cv::DescriptorExtractor::create(extractor_class);

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> kps(keypoints);
    extractor->compute(image, kps, descriptors);

    if (kps.size() == keypoints.size())
        return descriptors;

    // if the extractor has removed any keypoints, then we have
    // to copy each descriptor in turn, skipping the descriptor without
    // a keypoint. this will result in a descriptor with zero data in
    // the descriptor
    cv::Mat new_descriptors = cv::Mat::zeros(int(keypoints.size()), descriptors.cols, descriptors.type());
    copy_keypoint_descriptors(descriptors, kps, new_descriptors, keypoints);
    return new_descriptors;
}

inline
cv::Mat extract_regions(
    char const *                  const  extractor_class,
    std::vector<std::vector<cv::Point>> &regions,
    cv::Mat                       const &image)
{
    return extract_keypoints(extractor_class, to_keypoints(regions), image);
}

inline
cv::Mat save(char const * const pathname, cv::Mat const &image)
{
    return imwrite(pathname, image)? image : cv::Mat();
}

inline
cv::Mat const &show(char const * const window_name, cv::Mat const &image)
{
    imshow(window_name, image);
    return image;
}

}   // namespace detail

}   // namespace opencv_pipeline
