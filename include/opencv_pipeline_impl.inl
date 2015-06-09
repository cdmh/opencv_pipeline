#pragma once

namespace opencv_pipeline {

inline
cv::Mat load(char const * const pathname)
{
    return cv::imread(pathname);
}

inline
cv::Mat load(std::string const &pathname)
{
    return cv::imread(pathname);
}

inline
std::function<cv::Mat (cv::Mat const &)>
save(char const * const pathname)
{
    return std::bind(detail::save, pathname, std::placeholders::_1);
}

inline
cv::Rect roi(cv::Mat const &image)
{
    cv::Point tl;
    cv::Size  size;
    image.locateROI(size, tl);
    return cv::Rect(tl, size);
}

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


//
// operators -- these are not forward referenced in the
// header file as they are the mechanics used implicitly
// by the user rather than explicitly called functions
//

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

// some OpenCV functions return MatExpr
inline
cv::Mat operator|(cv::Mat const &left, cv::MatExpr(*right)(cv::Mat const &))
{
    return right(left);
}

// load an image with optional verification.
// whether or not to verify is compulsory -- verify or noverify must
// go between a load and any subsequent manipulations through the
// pipeline interface
inline
cv::Mat operator|(char const * const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::file_not_found(pathname);
    return image;
}

inline
cv::Mat operator|(std::string const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::file_not_found(pathname.c_str());
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


//
// image manipulation
//

inline
std::function<cv::Mat (cv::Mat const &)>
color_space(int code)
{
    using namespace std::placeholders;
    return std::bind(detail::color_space, _1, code);
}

inline
std::function<cv::Mat (cv::Mat const &)>
convert(int type)
{
    using namespace std::placeholders;
    return std::bind(detail::convert, _1, type);
}

inline
std::function<cv::Mat (cv::Mat const &)>
dilate(int dx, int dy)
{
    using namespace std::placeholders;
    return std::bind(detail::dilate, _1, dx, dy);
}

inline
std::function<cv::Mat (cv::Mat const &)>
erode(int dx, int dy)
{
    using namespace std::placeholders;
    return std::bind(detail::erode, _1, dx, dy);
}

inline
std::function<cv::Mat (cv::Mat const &)>
gaussian_blur(int dx, int dy, double sigmaX=0.0, double sigmaY=0.0, int border=cv::BORDER_DEFAULT)
{
    using namespace std::placeholders;
    return std::bind(detail::gaussian_blur, _1, dx, dy, sigmaX, sigmaY, border);
}

inline
cv::Mat gray(cv::Mat const &image)
{
    return image | color_space(cv::COLOR_BGR2GRAY);
}

inline
cv::Mat equalize_hist(cv::Mat image)
{
    cv::equalizeHist(image, image);
    return image;
}



inline
cv::Mat gray_bgr(cv::Mat image)
{
    return image | gray | color_space(cv::COLOR_GRAY2BGR);
}

inline
cv::Mat mirror(cv::Mat const &image)
{
    cv::Mat dst;
    flip(image, dst, 1);
    return dst;
}

inline
std::function<cv::Mat (cv::Mat const &)>
sobel(int dx, int dy, int ksize=3, double scale=1, double delta=0, int border=cv::BORDER_DEFAULT)
{
    using namespace std::placeholders;
    return std::bind(detail::sobel, _1, dx, dy, ksize, scale, delta, border);
}

inline
std::function<cv::Mat (cv::Mat const &)>
subtract(cv::Mat const &other)
{
    using namespace std::placeholders;
    return std::bind(detail::subtract, _1, other);
}

inline
std::function<cv::Mat (cv::Mat const &)>
threshold(double thresh, double maxval, int type=CV_THRESH_BINARY | CV_THRESH_OTSU)
{
    using namespace std::placeholders;
    return std::bind(detail::threshold, _1, thresh, maxval, type);
}



class video_pipeline
{
  public:
    video_pipeline(int device)
    {
       capture_.open(device);
    }

    video_pipeline(std::string const &pathname)
    {
       capture_.open(pathname);
    }

    video_pipeline(char const * const pathname)
      : video_pipeline(std::string(pathname))
    {
    }

    cv::Mat run()
    {
        cv::Mat image;
        capture_ >> image;
        if (image.empty())
            throw exceptions::end_of_file();
        return image;
    }

    video_pipeline(video_pipeline &&)                 = delete;
    video_pipeline &operator=(video_pipeline &&)      = delete;
    video_pipeline(video_pipeline const &)            = delete;
    video_pipeline &operator=(video_pipeline const &) = delete;

  private:
    cv::VideoCapture capture_;
};

// capture video from a file
inline
video_pipeline
video(char const * const pathname)
{
    return video_pipeline(pathname);
}

// capture video from a file
inline
video_pipeline
video(std::string const pathname)
{
    return video_pipeline(pathname);
}

// capture video from a camera device
inline
video_pipeline
camera(int device)
{
    return video_pipeline(device);
}

inline
std::pair<
    video_pipeline &,
    std::function<cv::Mat (cv::Mat const &)>>
operator|(
    video_pipeline &lhs,
    std::function<cv::Mat (cv::Mat const &)> rhs)
{
    return {lhs,rhs};
}

template<typename LHS, typename RHS>
std::pair<
    std::pair<LHS, RHS>,
    std::function<cv::Mat (cv::Mat const &)>>
operator|(
    std::pair<LHS, RHS> lhs, 
    std::function<cv::Mat (cv::Mat const &)> rhs)
{
    return {lhs, rhs};
}

template<typename LHS, typename RHS>
std::pair<
    std::pair<LHS, RHS>,
    std::function<cv::Mat (cv::Mat const &)>>
operator|(
    std::pair<LHS, RHS> lhs, 
    cv::Mat(*rhs)(cv::Mat const &))
{
    return {lhs, rhs};
}

typedef
enum { play }
terminator;

template<typename LHS, typename RHS>
cv::Mat run(std::pair<LHS, RHS> chain)
{
    return chain.second(run(chain.first));
}

cv::Mat run(video_pipeline &pipeline)
{
    return pipeline.run();
}

#pragma warning(push)
#pragma warning(disable: 4127)  // C4127 conditional expression is constant
template<typename LHS, typename RHS>
bool const
operator|(std::pair<LHS, RHS> lhs, terminator)
{
    try
    {
        while (1)
            run(lhs);
    }
    catch (exceptions::end_of_file &)
    {
        return true;
    }
    return false;
}
#pragma warning(pop)

}   // namespace opencv_pipeline
